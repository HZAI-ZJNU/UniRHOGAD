# unirhogad_project/data/preprocess.py

import torch
import dgl
import os
import argparse
from tqdm import tqdm
import numpy as np
from functools import partial

from dataset_loader import UniGADDataset, MRQSampler
from anomaly_generator import AnomalyGenerator

def get_preprocess_args():
    """
    为预处理脚本解析命令行参数。
    """
    parser = argparse.ArgumentParser("Uni-RHO-GAD Data Pre-processing Script")
    
    # --- 核心参数 ---
    parser.add_argument('--dataset', type=str, required=True, 
                        help="Name of the single-graph dataset (e.g., 'reddit', 'yelp')")
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Root directory where data is stored (relative to this script)')
    parser.add_argument('--save_dir', type=str, default='./data/preprocessed',
                        help='Directory to save the pre-computed data')
    
    # --- 新增：多副本增强控制参数 ---
    parser.add_argument('--num_aug_sets', type=int, default=1,
                        help="Number of different artificial anomaly sets to pre-compute.")

    # --- 采样器参数 ---
    parser.add_argument('--sp_type', type=str, default='star+norm', 
                        help="Subgraph sampling strategy (e.g., 'star+norm', 'khop')")
    parser.add_argument('--khop', type=int, default=1, 
                        help="Number of hops for k-hop based samplers")

    # --- 异常生成参数 ---
    parser.add_argument('--generate_anomalies', action='store_true', 
                        help="Flag to enable generation of artificial anomalies.")
    parser.add_argument('--aug_ratio', type=float, default=0.5, 
                        help="Ratio of normal samples to generate anomaly versions for.")
    # 节点增强参数
    parser.add_argument('--aug_node_num_perturb_edges', type=int, default=5, 
                        help="Number of perturbed edges for each artificial node anomaly.")
    parser.add_argument('--aug_node_feature_mix_ratio', type=float, default=0.5, 
                        help="Mixing ratio for node feature perturbation (beta).")
    # 边增强参数
    parser.add_argument('--aug_edge_feature_mix_ratio', type=float, default=0.5,
                        help="Mixing ratio for edge endpoint feature perturbation.")

    return parser.parse_args()

def preprocess_data(args):
    """
    主预处理函数。
    """
    print(f"--- Starting pre-processing for dataset: {args.dataset} ---")
    
    # 1. 加载原始数据集
    # sp_type 和 khop 暂时不传给Dataset，因为我们在这里手动创建Sampler
    dataset = UniGADDataset(name=args.dataset, data_dir=args.data_dir)
    
    if not dataset.is_single_graph:
        print(f"Dataset '{args.dataset}' is a multi-graph dataset. Pre-computation is not required. Skipping.")
        return

    original_graph = dataset.graph_list[0]
    sampler = MRQSampler(strategy=args.sp_type)
    original_features = original_graph.ndata['feature']

    # 2. 为所有原始节点和边预计算子图
    precomputed_data = {
        'node_subgraphs': {},
        'edge_subgraphs': {}
    }
    
    print(f"--- Pre-computing subgraphs for {original_graph.num_nodes()} nodes... ---")
    for node_id in tqdm(range(original_graph.num_nodes()), desc="Node Sampling"):
        subgraph = sampler.sample(original_graph, node_id, khop=args.khop)
        # 将原始特征存入子图，这样加载时就无需访问大图
        subgraph.ndata['feature'] = original_features[subgraph.ndata[dgl.NID]]
        precomputed_data['node_subgraphs'][node_id] = subgraph

    if 'edge_label' in original_graph.edata:
        print(f"--- Pre-computing subgraphs for {original_graph.num_edges()} edges... ---")
        src, dst = original_graph.edges()
        for edge_id in tqdm(range(original_graph.num_edges()), desc="Edge Sampling"):
            # 使用简单的1-hop邻域作为边子图
            u, v = src[edge_id], dst[edge_id]
            nodes_for_edge = torch.unique(torch.cat([u.unsqueeze(0), v.unsqueeze(0)]))
            # 注意：khop_nodes 可能会返回大量的节点，导致子图很大
            # 一个更可控的方法是只包含端点
            subgraph = dgl.node_subgraph(original_graph, nodes_for_edge, store_ids=True)
            subgraph.ndata['feature'] = original_features[subgraph.ndata[dgl.NID]]
            precomputed_data['edge_subgraphs'][edge_id] = subgraph

    # 3. (可选) 生成人工异常并为其计算子图
    if args.generate_anomalies:
        print("--- Generating artificial anomalies and their subgraphs... ---")
        anomaly_generator = AnomalyGenerator(original_graph) # 使用Dataset中已初始化的实例
        
        for set_idx in range(args.num_aug_sets):
            print(f"\n--- Generating augmentation set {set_idx + 1}/{args.num_aug_sets}... ---")
            # --- 节点异常 ---
            if 'node_label' in original_graph.ndata:
                normal_node_ids = (original_graph.ndata['node_label'] == 0).nonzero(as_tuple=True)[0]
                num_to_generate = int(len(normal_node_ids) * args.aug_ratio)
                # 每次循环都重新随机选择，保证多样性
                nodes_to_perturb = normal_node_ids[torch.randperm(len(normal_node_ids))[:num_to_generate]]
                
                aug_node_ids, perturbed_feats, perturbed_graph = anomaly_generator.generate_for_nodes(
                    nodes_to_perturb, args.aug_node_num_perturb_edges, args.aug_node_feature_mix_ratio
                )
                
                # 使用带索引的键名来保存
                aug_node_key = f'aug_node_subgraphs_set_{set_idx}'
                precomputed_data[aug_node_key] = {}
                
                if aug_node_ids is not None:
                    print(f"  - Pre-computing subgraphs for {len(aug_node_ids)} artificial node anomalies...")
                    full_perturbed_features = original_features.clone()
                    perturbed_map = {nid.item(): feat for nid, feat in zip(aug_node_ids, perturbed_feats)}
                    for i, feat in perturbed_map.items():
                        full_perturbed_features[i] = feat

                    for aug_node_id in tqdm(aug_node_ids, desc=f"Art. Node Sampling Set {set_idx+1}"):
                        subgraph = sampler.sample(perturbed_graph, aug_node_id.item(), khop=args.khop)
                        subgraph.ndata['feature'] = full_perturbed_features[subgraph.ndata[dgl.NID]]
                        precomputed_data[aug_node_key][aug_node_id.item()] = subgraph
        
            # ... 类似地，可以为正常边生成异常版本并计算子图 ...
            if 'edge_label' in original_graph.edata:
                normal_edge_ids = (original_graph.edata['edge_label'] == 0).nonzero(as_tuple=True)[0]
                num_to_generate = int(len(normal_edge_ids) * args.aug_ratio)
                edges_to_perturb = normal_edge_ids[torch.randperm(len(normal_edge_ids))[:num_to_generate]]

                new_edges, perturbed_ids, perturbed_feats = anomaly_generator.generate_for_edges(
                    edges_to_perturb, args.aug_edge_feature_mix_ratio
                )

                # 使用带索引的键名来保存
                aug_edge_key = f'aug_edge_subgraphs_set_{set_idx}'
                precomputed_data[aug_edge_key] = {}
                start_aug_edge_id = original_graph.num_edges() + (len(normal_edge_ids) * set_idx) # 确保多套set的ID不冲突

                if new_edges is not None:
                    print(f"  - Pre-computing subgraphs for {len(new_edges)} artificial edge anomalies...")
                    full_perturbed_features = original_features.clone()
                    perturbed_map = {nid.item(): feat for nid, feat in zip(perturbed_ids, perturbed_feats)}
                    for i, feat in perturbed_map.items():
                        full_perturbed_features[i] = feat

                    for i, (u, v_prime) in enumerate(tqdm(new_edges, desc=f"Art. Edge Sampling Set {set_idx+1}")):
                        aug_edge_id = start_aug_edge_id + i
                        nodes_for_edge = torch.unique(torch.tensor([u, v_prime]))
                        subgraph = dgl.node_subgraph(original_graph, nodes_for_edge, store_ids=True)
                        subgraph.ndata['feature'] = full_perturbed_features[subgraph.ndata[dgl.NID]]
                        precomputed_data[aug_edge_key][aug_edge_id] = subgraph

    # 4. 保存预计算结果
    os.makedirs(args.save_dir, exist_ok=True)
    # 构建一个与UniGAD类似的有意义的文件名
    filename_parts = [
        args.dataset,
        f"khop_{args.khop}",
        f"sptype_{args.sp_type.replace('+', '_')}"
    ]
    if args.generate_anomalies:
        filename_parts.append(f"aug_{args.aug_ratio}_sets_{args.num_aug_sets}")
        
    filename = ".".join(filename_parts) + ".pt"
    save_path = os.path.join(args.save_dir, filename)
    
    print(f"--- Saving pre-computed data to: {save_path} ---")
    torch.save(precomputed_data, save_path)
    print("--- Pre-processing finished successfully! ---")

if __name__ == '__main__':
    import sys
    # 将项目根目录添加到路径中
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.dataset_loader import UniGADDataset, MRQSampler
    from data.anomaly_generator import AnomalyGenerator

    args = get_preprocess_args()
    preprocess_data(args)