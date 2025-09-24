import torch
import yaml
import torch.nn as nn
from tqdm import tqdm
import os
import argparse
from argparse import Namespace
import pandas as pd
import time
from functools import partial
from data.dataset_loader import UniGADDataset, collate_fn_unify
from models.pretrain_model import GraphMAE_PAA
from models.ablation_models import Uni_RHO_GAD_Predictor_BaseGNN
from predictors.unirhogad_predictor import Uni_RHO_GAD_Predictor, Uni_RHO_GAD_Predictor_NoGNA
from e2e_trainer import Trainer
from utils.misc import set_seed
from torch.utils.data import WeightedRandomSampler

def get_args():
    """解析所有命令行参数"""
    parser = argparse.ArgumentParser("Uni-RHO-GAD End-to-End Training")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file for the experiment.")
    # (可选) 允许命令行覆盖个别关键参数
    parser.add_argument('--device', type=str, default=None, help="Override the device setting in the config file.")
    parser.add_argument('--epochs', type=int, default=None, help="Override the epochs setting in the config file.")

    # 消融实验相关参数
    parser.add_argument('--ablation', type=str, default=None, 
                    choices=['no_gna'], help="Specify which ablation study to run.")

    parser.add_argument('--baseline', type=str, default=None, 
                    choices=['gcn', 'gin'], # 您可以根据您实现的 GNN 扩展这个列表
                    help="Run a classic GNN baseline instead of the full model.")
    return parser.parse_args()


def main(args):

    # 1. 解析命令行参数，主要是获取配置文件路径
    cmd_args = get_args()

    # 2. 加载配置文件
    with open(cmd_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 3. 将字典转换为命名空间对象，方便以 `args.key` 的形式访问
    args = Namespace(**config_dict)
    
    # 4. (可选) 让命令行的个别参数覆盖配置文件中的值
    if cmd_args.device is not None:
        args.device = cmd_args.device
    if cmd_args.epochs is not None:
        args.epochs = cmd_args.epochs

    try:
        # 对所有可能使用科学记数法的浮点数参数进行转换
        args.lr = float(args.lr)
        args.pretrain_lr = float(args.pretrain_lr)
        args.l2 = float(args.l2)
    except (ValueError, TypeError) as e:
        print(f"Error: Failed to convert one or more config parameters to their expected numeric types.")
        print(f"Please check your YAML file for correct formatting. Original error: {e}")
        return # 转换失败则直接退出

    """主执行函数"""
    set_seed(args.seed)
    
    # 1. ==================== 数据加载 ====================
    print(f"--- Loading dataset: {args.dataset} ---")
    dataset = UniGADDataset(name=args.dataset, data_dir=args.data_dir)
    
    # 为每个划分创建 DataLoader
    # 我们只在一个 trial 上运行，所以 trial_id=0
    dataset.prepare_split(trial_id=0, seed=args.seed)
    train_subset = dataset.get_subset('train', trial_id=0)
    val_subset = dataset.get_subset('val', trial_id=0)
    test_subset = dataset.get_subset('test', trial_id=0)
    
    anomaly_generator_instance = None
    if args.use_anomaly_generation and dataset.is_single_graph:
        # 从 dataset 中获取已经初始化好的 generator
        anomaly_generator_instance = dataset.anomaly_generator

    collate_with_aug = partial(
        collate_fn_unify, 
        sampler=dataset.sampler, 
        anomaly_generator=anomaly_generator_instance,
        aug_ratio=args.aug_ratio,
        num_perturb_edges=args.aug_num_perturb_edges,
        feature_mix_ratio=args.aug_feature_mix_ratio,
        use_node_aug=args.use_node_aug,
        use_edge_aug=args.use_edge_aug
    )

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_aug)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_aug)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_aug)

    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}, Test samples: {len(test_subset)}")

    # 2. ==================== 模型构建 ====================
    print("--- Building models ---")
    # a. 加载预训练的GraphMAE_PAA模型 (仅用于特征提取)
    # 注意：这里的参数需要与你预训练时使用的参数大致匹配
    print(f"Instantiating pretrained model architecture: enc={args.pretrain_encoder_type}, dec={args.pretrain_decoder_type}, hid={args.pretrain_hid_dim}")
    pretrain_model = GraphMAE_PAA(
        in_dim=dataset.in_dim,
        hid_dim=args.pretrain_hid_dim,
        encoder_num_layer=args.pretrain_encoder_num_layer,
        decoder_num_layer=args.pretrain_decoder_num_layer,
        encoder_type=args.pretrain_encoder_type,
        decoder_type=args.pretrain_decoder_type
    )
    print(f"Loading pretrained weights from {args.pretrain_path}")
    pretrain_model.load_state_dict(torch.load(args.pretrain_path, map_location=args.device, weights_only=True))
    pretrain_model.eval()  # 切换到评估模式

    # 创建特征适配器，以处理预训练输出维度和主模型输入维度不匹配的情况
    pretrain_output_dim = pretrain_model.embed_dim
    if pretrain_output_dim != args.hid_dim:
        print(f"Dimension mismatch: Adapting pretrained output from {pretrain_output_dim} to main model dim {args.hid_dim}.")
        feature_adapter = nn.Linear(pretrain_output_dim, args.hid_dim)
    else:
        feature_adapter = nn.Identity()

    # b. 构建我们的主模型 Uni_RHO_GAD_Predictor
    # 传入 cross_modes 列表
    cross_modes_list = args.cross_modes.split(',')
    all_tasks_list = list(args.all_tasks)
    # model = Uni_RHO_GAD_Predictor(
    #     pretrain_model=pretrain_model,
    #     feature_adapter=feature_adapter,
    #     is_single_graph=dataset.is_single_graph,
    #     embed_dims=args.hid_dim,
    #     num_classes=2, # 异常/正常
    #     all_tasks=all_tasks_list,
    #     cross_modes=cross_modes_list,
    #     base_gnn_layers=args.base_gnn_layers,
    #     final_mlp_layers=args.final_mlp_layers,
    #     gna_projection_dim=args.gna_proj_dim,
    #     dropout_rate=args.dropout,
    #     activation=args.activation,
    #     residual=args.residual,
    #     norm=args.norm
    # )

    model_args = {
        "pretrain_model": pretrain_model,
        "feature_adapter": feature_adapter,
        "is_single_graph": dataset.is_single_graph,
        "embed_dims": args.hid_dim,
        "num_classes": 2,
        "all_tasks": all_tasks_list,
        "cross_modes": cross_modes_list,
        "base_gnn_layers": args.base_gnn_layers,
        "final_mlp_layers": args.final_mlp_layers,
        "gna_projection_dim": args.gna_proj_dim,
        "dropout_rate": args.dropout,
        "activation": args.activation,
        "residual": args.residual,
        "norm": args.norm
    }
    # --- 核心修改：添加基线模型选择逻辑 ---
    if cmd_args.baseline:
        print(f"--- Running Baseline Experiment: {cmd_args.baseline.upper()} ---")
        # 将基线类型作为参数传入
        model = Uni_RHO_GAD_Predictor_BaseGNN(base_gnn_type=cmd_args.baseline, **model_args)
    elif cmd_args.ablation == 'no_gna':
        print("--- Running Ablation Study: No GNA ---")
        model = Uni_RHO_GAD_Predictor_NoGNA(**model_args)
    else:
        print("--- Running Full Model: Uni_RHO_GAD_Predictor ---")
        model = Uni_RHO_GAD_Predictor(**model_args)
    # --- 修改结束 ---

    # 3. ==================== 训练启动 ====================
    print("--- Initializing Trainer ---")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        args=args
    )
    
    print("--- Starting end-to-end training ---")
    final_test_metrics, total_time_cost = trainer.train()
    print("--- Training finished ---")

    # 4. ==================== 结果保存 ====================
    if final_test_metrics:
        all_results = []

        # 这些是所有实验都会保存的通用参数
        common_params = [
            'dataset', 'seed', 'pretrain_model', 'cross_mode',
            'hid_dim', 'base_gnn_layers', 'final_mlp_layers',
            'lr', 'l2', 'batch_size', 'epochs', 'patience',
            'w_one_class', 'w_gna', 'w_classification',
            'time_cost'
        ]
        
        # 这些是只在单图实验中保存的参数
        single_graph_params = [
            'use_anomaly_generation', 'use_node_aug', 'use_edge_aug',
            'aug_ratio', 'aug_num_perturb_edges', 'aug_feature_mix_ratio',
            'w_classification_n', 'w_classification_e' # 添加任务专属权重
        ]
        
        # 这些是只在多图实验中保存的参数
        multi_graph_params = [
            'use_downstream_multi_graph_aug',
            'aug_drop_node_rate', 'aug_perturb_edge_rate', 'aug_mask_feature_rate'
        ]

        # --- 步骤 2: 根据当前场景确定要保存的完整参数列表 ---
        if dataset.is_single_graph:
            params_to_save = common_params + single_graph_params
        else:
            params_to_save = common_params + multi_graph_params

        for mode, metrics_dict in final_test_metrics.items():
            result_row = {}
            
            # --- 步骤 3: 使用 getattr 安全地填充所有参数 ---
            for param in params_to_save:
                result_row[param] = getattr(args, param, None)
            
            # --- 步骤 4: 更新特定于当前循环的信息 (与之前相同) ---
            result_row['pretrain_model'] = os.path.basename(args.pretrain_path)
            result_row['cross_mode'] = mode.replace('_to_', '2') # 保持输出格式一致
            result_row['time_cost'] = total_time_cost / len(final_test_metrics) if final_test_metrics else total_time_cost

            # --- 步骤 5: 添加所有性能指标 (与之前相同) ---
            for task, metrics in metrics_dict.items():
                for metric_name, value in metrics.items():
                    result_row[f'{task}_{metric_name}'] = value
            
            all_results.append(result_row)
        
        results_df = pd.DataFrame(all_results)
        
        dataset_name_clean = args.dataset.replace('/', '_')
        timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
        filename = f"results_{dataset_name_clean}_{timestamp}.csv"
        save_path = os.path.join(args.results_dir, filename)
        os.makedirs(args.results_dir, exist_ok=True)
        
        results_df.to_csv(save_path, index=False)
        print("\n--- Final Test Results ---")
        # 使用 to_string() 打印完整的 DataFrame，避免列被省略
        print(results_df.to_string())
        print(f"\nResults saved to: {save_path}")
    else:
        print("Training finished, but no valid test metrics were generated.")
        
if __name__ == '__main__':
    args = get_args()
    main(args)