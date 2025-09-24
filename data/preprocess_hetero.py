# unirhogad_project/preprocess_hetero.py

import dgl
import torch
import numpy as np
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix, csc_matrix
import os
import argparse
import pandas as pd # 导入 pandas

def preprocess_and_save(args):
    """
    执行完整的预处理流程：从 .mat 文件加载，净化，并保存为 DGL .bin 文件。
    """
    # 1. 根据数据集名称确定文件名并加载 .mat 文件
    if args.dataset == 'amazon':
        filename = 'Amazon.mat'
    elif args.dataset == 'yelp':
        filename = 'YelpChi.mat'
    else:
        raise ValueError("This script is designed for 'amazon' or 'yelp' dataset.")

    mat_path = os.path.join(args.data_dir, args.dataset, filename)
    print(f"--- Loading raw data from: {mat_path} ---")
    try:
        data = sio.loadmat(mat_path)
    except FileNotFoundError:
        print(f"Error: .mat file not found at {mat_path}.")
        return

    # 2. 从 .mat 文件中提取核心数据
    node_features = torch.tensor(data['features'].toarray(), dtype=torch.float)
    node_labels = torch.tensor(data['label'].flatten(), dtype=torch.long)
    num_nodes = node_features.shape[0]
    print(f"Loaded data for {num_nodes} nodes.")

    # 3. 根据数据集特性，区分强弱关系并抽取邻接矩阵
    print("--- Extracting and processing meta-path relations ---")
    strong_adj = csc_matrix((num_nodes, num_nodes), dtype=np.float32)
    weak_adjs = []
    weak_adj_names = []

    if args.dataset == 'yelp':
        weak_adjs.extend([data['net_rur'], data['net_rsr'], data['net_rtr']])
        weak_adj_names.extend(['net_rur', 'net_rsr', 'net_rtr'])
        print("  - Identified weak relations for Yelp: net_rur, net_rsr, net_rtr")
    elif args.dataset == 'amazon':
        weak_adjs.extend([data['net_upu'], data['net_usu'], data['net_uvu']])
        weak_adj_names.extend(['net_upu', 'net_usu', 'net_uvu'])
        print("  - Identified weak relations for Amazon: net_upu, net_usu, net_uvu")
    
    # 4. 对关系进行剪枝和加权融合
    print("--- Pruning relations based on feature similarity ---")
    final_adj = strong_adj.astype(np.float32) * args.w_strong
    if final_adj.nnz > 0:
        print(f"  - Started with {final_adj.nnz} edges from strong relations.")

    for name, weak_adj in zip(weak_adj_names, weak_adjs):
        if weak_adj.nnz == 0: 
            print(f"  - Skipping empty relation: {name}")
            continue
        
        print(f"  - Processing relation '{name}' with {weak_adj.nnz} edges...")
        weak_adj_coo = weak_adj.tocoo()
        src, dst = weak_adj_coo.row, weak_adj_coo.col
        
        edge_features_src = node_features[src]
        edge_features_dst = node_features[dst]
        similarities = torch.nn.functional.cosine_similarity(edge_features_src, edge_features_dst)
        
        mask = similarities > args.pruning_threshold
        
        pruned_src = src[mask.numpy()]
        pruned_dst = dst[mask.numpy()]
        
        print(f"    - Pruned {weak_adj.nnz} edges down to {len(pruned_src)} edges.")
        
        if len(pruned_src) > 0:
            pruned_weights = similarities[mask].numpy() if args.use_similarity_weights else np.ones(len(pruned_src))
            pruned_adj = coo_matrix((pruned_weights, (pruned_src, pruned_dst)), shape=(num_nodes, num_nodes))
            final_adj += args.w_weak * pruned_adj.astype(np.float32)

    # 5. 构建最终的DGL同质图 (核心修复区域)
    print("--- Building final DGL graph ---")
    final_adj = final_adj.tocoo()
    
    
    # 步骤 a: 创建一个包含所有潜在边和权重的 DataFrame
    df = pd.DataFrame({
        'src': final_adj.row,
        'dst': final_adj.col,
        'weight': final_adj.data
    })
    
    # 步骤 b: 规范化边的表示，以便将 (u, v) 和 (v, u) 视为同一条边
    df['u'] = np.minimum(df['src'], df['dst'])
    df['v'] = np.maximum(df['src'], df['dst'])
    
    # 步骤 c: 按权重降序排序，然后对规范化的边去重，保留第一条（即权重最大的）
    print(f"  - Coalescing parallel edges... (before: {len(df)} edges)")
    df_simple = df.sort_values('weight', ascending=False).drop_duplicates(subset=['u', 'v'], keep='first')
    print(f"  - Coalescing finished. (after: {len(df_simple)} edges)")
    
    # 步骤 d: 从去重后的 DataFrame 中提取最终的边和权重
    src_simple = torch.tensor(df_simple['src'].values)
    dst_simple = torch.tensor(df_simple['dst'].values)
    weights_simple = torch.tensor(df_simple['weight'].values, dtype=torch.float)
    
    # 步骤 e: 使用净化后的边列表创建最终的图
    g_final = dgl.graph((src_simple, dst_simple), num_nodes=num_nodes)
    g_final.edata['weight'] = weights_simple
    
    # 步骤 f: 移除可能存在的自环
    g_final = dgl.remove_self_loop(g_final)
    
    
    g_final.ndata['feature'] = node_features
    g_final.ndata['node_label'] = node_labels

    # 6. 推断并添加边标签
    print("--- Inferring edge labels ---")
    src_final, dst_final = g_final.edges()
    src_labels = node_labels[src_final]
    dst_labels = node_labels[dst_final]
    edge_labels = src_labels * dst_labels
    g_final.edata['edge_label'] = edge_labels
    
    print(f"Final graph has {g_final.num_nodes()} nodes and {g_final.num_edges()} edges.")
    print(f"  - Anomaly nodes: {node_labels.sum().item()} ({(node_labels.sum()/num_nodes*100):.2f}%)")
    if g_final.num_edges() > 0:
        print(f"  - Anomaly edges: {edge_labels.sum().item()} ({(edge_labels.sum()/g_final.num_edges()*100):.2f}%)")
    else:
        print("  - Anomaly edges: 0")

    # 7. 保存处理好的图
    output_dir = os.path.join(args.output_dir, 'edge_labels')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{args.dataset}-els"
    save_path = os.path.join(output_dir, filename)
    dgl.save_graphs(save_path, [g_final])
    print(f"--- Preprocessing finished. Cleaned graph saved to: {save_path} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Heterogeneous Graph Preprocessing for Yelp and Amazon")
    parser.add_argument('--dataset', type=str, required=True, choices=['amazon', 'yelp'],
                        help="Name of the dataset to preprocess.")
    parser.add_argument('--data_dir', type=str, default='./data/hetero',
                        help="Directory where raw .mat files are stored.")
    parser.add_argument('--output_dir', type=str, default='./data/hetero',
                        help="Root directory to save the processed homogeneous graph.")
    
    parser.add_argument('--pruning_threshold', type=float, default=0.1,
                        help="Cosine similarity threshold for edge pruning. Edges below this are removed.")
    parser.add_argument('--w_strong', type=float, default=1.0,
                        help="Weight for strong relations during fusion.")
    parser.add_argument('--w_weak', type=float, default=0.5,
                        help="Weight for weak relations during fusion.")
    parser.add_argument('--use_similarity_weights', action='store_true',
                        help="If set, use cosine similarity as edge weights for pruned weak relations.")
    
    args = parser.parse_args()
    preprocess_and_save(args)