# unirhogad_project/e2e_trainer.py

import torch
import torch.nn.functional as F
import dgl.function as fn
from tqdm import tqdm
import numpy as np
import pprint 
import dgl
import pprint
import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from utils.misc import FocalLoss
from predictors.unirhogad_predictor import Uni_RHO_GAD_Predictor
from data.anomaly_generator import AnomalyGenerator
from models.pretrain_model import augment_graph_view
from utils.misc import get_current_lr

class Trainer:
    def __init__(self,  model, train_loader, val_loader, test_loader, args):
        """
        初始化训练器。

        Args:
            model: 主模型 Uni_RHO_GAD_Predictor。
            train_loader, val_loader, test_loader: 数据加载器。
            args: 命令行参数。
        """
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.device = args.device
        
        # 1. 从 args 中获取一个新参数来控制是否冻结 RHOEncoder
        freeze_rho_encoder = getattr(args, 'freeze_rho_encoder', False)

        # 2. 根据参数构建要优化的参数列表
        params_to_optimize = []
        if freeze_rho_encoder:
            print("--- [EXPERIMENT MODE] RHOEncoder is FROZEN. Only training fusion heads and centers. ---")
            # 只将 fusion_heads 和 centers 的参数加入优化列表
            params_to_optimize.extend(list(model.fusion_heads.parameters()))
            params_to_optimize.extend(list(model.centers.parameters()))
            # 注意：feature_adapter 通常也应该被训练
            params_to_optimize.extend(list(model.feature_adapter.parameters()))
        else:
            print("--- [NORMAL MODE] Training all model parameters (end-to-end). ---")
            # 训练所有参数
            params_to_optimize = model.parameters()

        # 3. 使用构建好的参数列表来初始化优化器
        self.optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.l2)

        # 初始化时不设置alpha，因为我们将在forward时动态传入
        self.classification_loss_fn = FocalLoss(gamma=2, alpha=None, reduction="mean")


    def _get_best_f1(self, labels, probs):
        """通过搜索阈值找到最佳的宏F1分数"""
        best_f1 = 0
        # 确保标签和概率是numpy数组
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        if isinstance(probs, torch.Tensor): probs = probs.cpu().numpy()
        
        for thres in np.linspace(0.05, 0.95, 19):
            preds = (probs > thres).astype(int)
            best_f1 = max(best_f1, f1_score(labels, preds, average='macro', zero_division=0))
        return best_f1

    def _compute_metrics(self, labels, probs):
        """计算所有评估指标"""
        # 确保标签和概率是numpy数组
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        if isinstance(probs, torch.Tensor): probs = probs.cpu().numpy()
        
        # 检查标签是否只有一个类别，这会导致AUROC计算错误
        if len(np.unique(labels)) < 2:
            print("Warning: Only one class present in labels. Metrics will be trivial.")
            return {'AUROC': 0.5, 'AUPRC': np.mean(labels), 'MacroF1': f1_score(labels, probs > 0.5, average='macro', zero_division=0)} # 返回一个无意义但安全的默认值
            
        return {
            'AUROC': roc_auc_score(labels, probs),
            'AUPRC': average_precision_score(labels, probs),
            'MacroF1': self._get_best_f1(labels, probs)
        }

    def _calculate_loss_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        根据一个批次的标签动态计算类别权重。
        权重 = 1 / (类别频率)
        """
        if labels.numel() == 0:
            return None

        # 计算每个类别的样本数
        class_counts = torch.bincount(labels, minlength=2).float()
        
        # 如果某个类别不存在，则其权重为0，避免除零错误
        # 权重计算：总样本数 / (类别数 * 类别样本数) 是一种常见的归一化方法
        # 或者更简单的：1 / 类别频率
        total_samples = class_counts.sum()
        if total_samples == 0:
            return None
        
        # 避免除以零
        class_freq = class_counts / total_samples
        weights = 1.0 / (class_freq + 1e-6) # 加一个小的epsilon防止除零
        
        # 归一化权重，使其和为类别数 (可选，但推荐)
        weights = weights / weights.sum() * 2.0
        
        return weights.to(self.device)

    def _calculate_composite_score(self, metrics_dict: dict) -> float:
        """
        根据一个epoch的完整验证集评估结果，计算一个综合分数。
        """
        # --- 策略：对所有我们关心的任务和指标，进行加权平均 ---
        scores = []
        weights = []
        
        # 遍历所有 cross_modes
        for mode, tasks_metrics in metrics_dict.items():
            # 遍历该模式下的所有任务
            for task, metrics in tasks_metrics.items():
                # 遍历我们关心的指标
                if 'AUPRC' in metrics:
                    # 我们可以给 AUPRC 更高的权重，因为它更重要
                    scores.append(metrics['AUPRC'])
                    weights.append(1.0) # AUPRC 权重为 1.0
                if 'AUROC' in metrics:
                    scores.append(metrics['AUROC'])
                    weights.append(0.5) # AUROC 权重为 0.5 (可以调整)

        if not scores:
            return -1.0 # 如果没有任何有效分数

        # 计算加权平均分
        composite_score = np.average(scores, weights=weights)
        return composite_score

    def train(self):
        """执行完整的训练、验证和早停流程"""
        best_composite_score = -1.0
        patience_counter = 0
        best_test_metrics = None
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            # --- 训练阶段 ---
            self.model.train()
            # 预训练模型始终处于评估模式
            self.model.pretrain_model.eval() 
            
            epoch_total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training", leave=False)

            # 初始化计时器
            time_profile = {
                'collate': [], 'to_device': [], 'forward': [], 
                'loss_calc': [], 'backward': [], 'step': []
            }
            
            t_epoch_start = time.time() # 记录每个批次开始的时间点

            for data in pbar:

                t_batch_start = time.time()

                # collate_fn 返回 (原始图/批处理图, 标签字典, 任务图字典)
                original_graph, batched_labels, task_graphs = data

                time_profile['collate'].append(time.time() - t_batch_start)
                t_after_collate = time.time()

                # 使用 getattr 来安全地访问属性，如果不存在，则默认为 False
                use_multi_graph_aug = getattr(self.args, 'use_downstream_multi_graph_aug', False)
                # 这个逻辑只在多图场景下，并且开关打开时执行
                if use_multi_graph_aug and not self.model.is_single_graph:
                    # 确认我们有图可以增强
                    if 'g' in task_graphs:
                        g_batched = task_graphs['g']
                        original_features = g_batched.ndata['feature']
                        
                        # 解批处理 -> 逐图增强 -> 重新批处理
                        graphs = dgl.unbatch(g_batched)
                        nodes_per_graph = g_batched.batch_num_nodes().tolist()
                        node_offsets = [0] + np.cumsum(nodes_per_graph).tolist()
                        
                        aug_graph_list = []
                        for i, g in enumerate(graphs):
                            start, end = node_offsets[i], node_offsets[i+1]
                            g_features = original_features[start:end]
                            # 为下游训练创建一个增强视图
                            # 这里的增强参数可以硬编码，或者也加入到配置文件中
                            aug_graph_list.append(augment_graph_view(g, g_features, 
                                                                     drop_node_rate=self.args.aug_drop_node_rate, 
                                                                     perturb_edge_rate=self.args.aug_perturb_edge_rate, 
                                                                     mask_feature_rate=self.args.aug_mask_feature_rate))
                        
                        # 用增强后的批处理图替换原来的图
                        task_graphs['g'] = dgl.batch(aug_graph_list)
                
                # 将所有数据移动到设备
                task_graphs = {k: v.to(self.device) for k, v in task_graphs.items()}
                batched_labels = {k: v.to(self.device) for k, v in batched_labels.items()}
                if original_graph:
                    original_graph = original_graph.to(self.device)

                time_profile['to_device'].append(time.time() - t_after_collate)
                t_after_to_device = time.time()

                
                # 准备 normal_masks 用于单类损失
                normal_masks = {k: (v == 0) for k, v in batched_labels.items() if v.numel() > 0}

                all_logits, shared_losses = self.model(
                    batched_inputs=task_graphs,
                    original_graph=original_graph,
                    normal_masks=normal_masks
                )

                time_profile['forward'].append(time.time() - t_after_to_device)
                t_after_forward = time.time()

                # 计算总损失
                loss_cls = 0
                # 遍历所有 cross_modes 的输出
                for mode_name, logits_dict in all_logits.items():
                    for task, logits in logits_dict.items():
                        # 确定用于监督的标签键
                        # 对于多图，所有任务都用图标签'g'监督
                        # 对于单图，每个任务用自己的标签'n'或'e'监督
                        label_key = 'g' if not self.model.is_single_graph else task

                        if label_key in batched_labels and batched_labels[label_key].numel() > 0:
                            labels = batched_labels[label_key]
                            # 确保logits和labels的批次维度匹配
                            if logits.shape[0] == labels.shape[0]:
                                class_weights = self._calculate_loss_weights(labels)
                                task_weight = getattr(self.args, f'w_classification_{task}', 1.0)
                                loss_cls += task_weight * self.classification_loss_fn(logits, labels, weight=class_weights)
                            else:
                                # 添加警告，防止未来出现未预料的维度不匹配
                                print(f"Warning: Skipping loss calculation for task '{task}' in mode '{mode_name}' due to shape mismatch. "
                                      f"Logits: {logits.shape}, Labels: {labels.shape}")
                            
                
                # 加权组合所有损失
                loss = (self.args.w_classification * loss_cls +
                        self.args.w_gna * shared_losses.get('gna', 0) +
                        self.args.w_one_class * shared_losses.get('one_class', 0))

                time_profile['loss_calc'].append(time.time() - t_after_forward)
                t_after_loss_calc = time.time()
                
                self.optimizer.zero_grad()
                loss.backward()

                time_profile['backward'].append(time.time() - t_after_loss_calc)
                t_after_backward = time.time()

                self.optimizer.step()
                
                time_profile['step'].append(time.time() - t_after_backward)

                epoch_total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            
            avg_train_loss = epoch_total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
            print(f"Epoch {epoch:03d} | Avg Train Loss: {avg_train_loss:.4f}")

            # --- 在每个 epoch 结束后打印平均耗时 ---
            print(f"Epoch {epoch:03d} | Avg Train Loss: {avg_train_loss:.4f}")
            print(f"--- Epoch {epoch:03d} Time Profile (Avg per batch) ---")
            for key, values in time_profile.items():
                avg_time = np.mean(values) if values else 0
                print(f"  - Avg {key.capitalize():<12}: {avg_time:.6f} s")
            print(f"  - Avg Total Batch Time: {np.mean([sum(times) for times in zip(*time_profile.values())]):.6f} s")
            print(f"  - Total Epoch Time:     {time.time() - t_epoch_start:.2f} s")
            print("-" * 40)


            # --- 验证阶段 ---
            val_metrics = self.evaluate('val')
            
            # --- 早停逻辑 ---
            # 1. 计算当前epoch的综合分数
            current_composite_score = self._calculate_composite_score(val_metrics)
            
            print(f"Epoch {epoch:03d} | Avg Train Loss: {avg_train_loss:.4f} | Composite Val Score: {current_composite_score:.4f}")

            # 2. 基于综合分数进行模型选择
            if current_composite_score > best_composite_score:
                best_composite_score = current_composite_score
                patience_counter = 0
                print("New best composite validation score! Evaluating on test set...")
                best_test_metrics = self.evaluate('test')

                
                # 遍历所有 cross_modes 的结果
                for mode, metrics_dict in best_test_metrics.items():
                    # 打印模式名称，例如 "> Test Results for mode [ne_to_ne]:"
                    print(f"  > Test Results for mode [{mode.replace('_to_', '2')}]:")
                    # 遍历该模式下所有任务的结果
                    for task, metrics in metrics_dict.items():
                        # 格式化输出每个任务的所有指标
                        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
                        print(f"    - Task [{task}]: {metrics_str}")
            else:
                patience_counter += 1
            
            if patience_counter >= self.args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break
        
        total_time_cost = time.time() - start_time
        print("\n--- Training Finished ---")
        return best_test_metrics, total_time_cost

    def evaluate(self, split='val'):
        self.model.eval()

        loader = self.val_loader if split == 'val' else self.test_loader

        
        # 初始化用于存储所有预测结果的结构
        all_preds = {mode: {task: [] for task in head.output_route} for mode, head in self.model.fusion_heads.items()}
        all_labels = {mode: {task: [] for task in head.output_route} for mode, head in self.model.fusion_heads.items()}


        with torch.no_grad():
            for data in tqdm(loader, desc=f"Evaluating on {split} set", leave=False):
                
                original_graph, batched_labels, batched_inputs = data
                batched_inputs = {k: v.to(self.device) for k, v in batched_inputs.items()}
                
                if original_graph:
                    original_graph = original_graph.to(self.device)

                all_logits, _ = self.model(
                    batched_inputs=batched_inputs,
                    original_graph=original_graph
                    )


                for mode_name, logits_dict in all_logits.items():
                    for task, logits in logits_dict.items():
                        label_key = 'g' if not self.model.is_single_graph else task

                        if label_key in batched_labels and batched_labels[label_key].numel() > 0:
                            labels = batched_labels[label_key].cpu()
                            # 确保logits和labels的批次维度匹配
                            if logits.shape[0] == labels.shape[0]:
                                probs = F.softmax(logits, dim=1)[:, 1].cpu()
                                if mode_name in all_preds and task in all_preds[mode_name]:
                                    all_preds[mode_name][task].append(probs)
                                    all_labels[mode_name][task].append(labels)
        
        # 计算最终指标
        final_metrics = {}
        for mode_name in all_preds:
            final_metrics[mode_name] = {}
            for task in all_preds[mode_name]:
                if all_preds[mode_name][task]:
                    preds_cat = torch.cat(all_preds[mode_name][task])
                    labels_cat = torch.cat(all_labels[mode_name][task])
                    final_metrics[mode_name][task] = self._compute_metrics(labels_cat, preds_cat)

        return final_metrics