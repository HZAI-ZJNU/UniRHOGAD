import dgl
import torch
import os
from tqdm import tqdm

def generate_edge_labels(dataset_names, raw_data_dir, save_dir):
    """
    为所有同质单图数据集生成带有多层级标签的 .bin 文件。
    采用 UniGAD 的宽松策略：只要有一端是异常，边即为异常。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for ds_name in tqdm(dataset_names, desc="Processing Homogeneous Datasets"):
        # 1. 构建原始数据集的路径
        # 假设原始文件直接存放在 raw_dataset 目录下，例如 reddit.bin
        path = os.path.join(raw_data_dir, ds_name) 
        
        try:
            # 假设原始文件是 DGL .bin 格式
            graph = dgl.load_graphs(path)[0][0]
        except Exception as e:
            print(f"\nCould not load graph for {ds_name} from {path}. Error: {e}. Skipping.")
            continue
        
        # 2. 预处理节点标签
        if 'label' not in graph.ndata:
            print(f"\n'label' not found in ndata for {ds_name}. Skipping.")
            continue
        
        node_labels = graph.ndata['label']
        # 重命名节点标签字段
        graph.ndata['node_label'] = node_labels
        if 'label' in graph.ndata: # 确保删除
            del graph.ndata['label']
        
        # 3. 生成边标签
        u, v = graph.edges()
        labels_u = node_labels[u]
        labels_v = node_labels[v]
        
        # 旧的、严格的规则
        # edge_labels = ((labels_u + labels_v) / 2.0).round().long() # 等价于 label_u * label_v
        
        # 新的、宽松的规则 (UniGAD 思想)
        edge_labels = torch.max(labels_u, labels_v)
        
        # 4. 添加边标签到图数据
        graph.edata['edge_label'] = edge_labels
        
        # 5. 保存处理后的图
        save_path = os.path.join(save_dir, f"{ds_name}-els")
        dgl.save_graphs(save_path, [graph])

        print(f"\nFinished processing {ds_name}. Saved to {save_path}")
        print(f"  - Total Edges: {graph.num_edges()}")
        print(f"  - Anomaly Edges (using max rule): {edge_labels.sum().item()} ({(edge_labels.sum().item() / graph.num_edges() * 100):.2f}%)")

if __name__ == "__main__":
    # 定义您的项目根目录
    PROJECT_ROOT = "/home/pangu/ljc/unirhogad_project/UniRHOGAD"
    
    # 定义原始数据集和目标保存目录
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data/raw_dataset")
    SAVE_DIR = os.path.join(PROJECT_ROOT, "data/edge_labels")
    
    # 定义需要处理的同质单图数据集列表
    homogeneous_datasets = [
        # 'reddit', 
        # 'weibo', 
        'tfinance', 
        # 'tolokers', 
        # 'questions'
    ]
    
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Save directory for processed files: {SAVE_DIR}")
    
    generate_edge_labels(homogeneous_datasets, RAW_DATA_DIR, SAVE_DIR)