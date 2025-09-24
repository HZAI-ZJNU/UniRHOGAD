import torch
import dgl
import numpy as np
import networkx as nx
import community as community_louvain
from typing import Tuple, Optional, List

class AnomalyGenerator:
    def __init__(self, g: dgl.DGLGraph):
        """
        初始化异常生成器。
        
        Args:
            g (dgl.DGLGraph): 原始的、完整的图。
        """
        print("Initializing Anomaly Generator...")
        self.graph = g
        self.num_nodes = g.num_nodes()
        self.features = g.ndata['feature']
        
        # 1. 执行社区发现 (一次性预处理)
        print("  - Running Louvain community detection...")

        # 确保图在CPU上
        g_cpu = g.to('cpu')
        
        # 步骤 a: 创建一个空的无向 NetworkX 图
        nx_g = nx.Graph()
        
        # 步骤 b: 添加 DGL 图中的所有节点
        nx_g.add_nodes_from(range(g_cpu.num_nodes()))
        
        # 步骤 c: 添加 DGL 图中的所有边
        # g_cpu.edges() 返回源节点和目标节点的元组
        src, dst = g_cpu.edges()
        # 将 tensor 转换为 numpy 数组以便迭代
        edges_to_add = list(zip(src.numpy(), dst.numpy()))
        nx_g.add_edges_from(edges_to_add)
        
        # 现在 nx_g 是一个保证无向的 NetworkX 图对象
        
        partition = community_louvain.best_partition(nx_g)
        
        # 将社区划分结果存为tensor
        communities = torch.zeros(self.num_nodes, dtype=torch.long)
        for node, com_id in partition.items():
            communities[node] = com_id
        
        self.communities = communities.to(g.device)
        self.num_communities = communities.max().item() + 1
        print(f"  - Found {self.num_communities} communities.")

    def generate_for_nodes(self, node_ids: torch.Tensor, num_perturb_edges: int = 5, feature_mix_ratio: float = 0.5):
        """
        为一批给定的节点ID生成特征与结构联合扰动的异常版本。
        
        Args:
            node_ids (torch.Tensor): 要进行扰动的正常节点的ID列表。
            num_perturb_edges (int): 为每个节点添加的异常边的数量。
            feature_mix_ratio (float): 特征混合比例 (beta)。

        Returns:
            (torch.Tensor, dgl.DGLGraph): 扰动后的新特征，以及一个包含新边的图（用于采样）。
                                          注意：这里返回的是一个修改了边的图，而不是子图。
        """
        device = self.graph.device
        node_ids = node_ids.to(device)
        
        
        new_src_nodes, new_dst_nodes = [], []
        perturbed_features_list = []
        valid_perturbed_node_ids = []

        for node_id in node_ids:
            node_id_item = node_id.item()
            original_community = self.communities[node_id_item]
            
            # --- 1. 结构扰动 (Structure Perturbation) ---
            # 随机选择一个遥远社区
            target_community = original_community
            while target_community == original_community:
                target_community = np.random.randint(0, self.num_communities)
            
            # 找到目标社区的所有节点
            nodes_in_target_community = (self.communities == target_community).nonzero(as_tuple=True)[0]
            
            if len(nodes_in_target_community) == 0:
                continue # 如果目标社区为空，跳过

            # 从目标社区中随机选择几个节点作为新的“异常邻居”
            num_to_sample = min(num_perturb_edges, len(nodes_in_target_community))
            new_neighbors = nodes_in_target_community[torch.randperm(len(nodes_in_target_community))[:num_to_sample]]
            
            # 记录新的异常边
            new_src_nodes.extend([node_id] * num_to_sample)
            new_dst_nodes.extend(new_neighbors)

            # --- 2. 特征扰动 (Feature Perturbation) ---
            original_feature = self.features[node_id]
            neighbor_features = self.features[new_neighbors]
            
            # 计算新邻居的平均特征
            mean_neighbor_feature = neighbor_features.mean(dim=0)
            
            # 混合原始特征和邻居特征
            perturbed_feature = (1 - feature_mix_ratio) * original_feature + feature_mix_ratio * mean_neighbor_feature
            perturbed_features_list.append(perturbed_feature)
            # 记录下这个成功被扰动的节点ID
            valid_perturbed_node_ids.append(node_id)

        if not perturbed_features_list:
            # 如果没有生成任何异常，返回空
            return None, None, None

        # --- 3. 构建返回结果 ---
        # a. 将所有扰动后的特征堆叠成一个张量
        perturbed_features = torch.stack(perturbed_features_list)
        
        # b. 创建一个包含原始边和新扰动边的图
        perturbed_graph = self.graph.clone()

        if new_src_nodes: # 确保列表不为空
            # 1. 获取目标图期望的ID类型
            target_dtype = perturbed_graph.idtype
            
            # 2. 使用检测到的类型来创建张量
            src_tensor = torch.tensor(new_src_nodes, device=device, dtype=target_dtype)
            dst_tensor = torch.tensor(new_dst_nodes, device=device, dtype=target_dtype)
            
            perturbed_graph.add_edges(src_tensor, dst_tensor)
        
        return torch.stack(valid_perturbed_node_ids), perturbed_features, perturbed_graph

    def generate_for_edges(self, 
                           edge_ids: torch.Tensor, 
                           feature_mix_ratio: float = 0.5) -> Optional[Tuple[List[int], torch.Tensor, dgl.DGLGraph]]:
        """
        为一批给定的正常边ID生成人工异常版本。
        策略：保持源节点u不变，将目标节点v重定向到遥远社区的v'，然后混合u和v'的特征来伪装u。
        
        Args:
            edge_ids (torch.Tensor): 要扰动的正常边的ID列表。
            feature_mix_ratio (float): 特征混合比例。

        Returns:
            Optional[Tuple[...]]
                - new_edges_list (List[Tuple[int, int]]): 新生成的异常边的端点列表 [(u, v'), ...]
                - perturbed_node_ids (torch.Tensor): 被修改了特征的节点的ID列表 (主要是u)
                - perturbed_node_features (torch.Tensor): 对应的新特征
        """
        device = self.graph.device
        edge_ids = edge_ids.to(device)
        
        # 1. 找到原始边的端点
        src_nodes, dst_nodes = self.graph.find_edges(edge_ids)
        
        new_edges_list = []
        perturbed_node_map = {} # 使用字典来存储被扰动的节点ID和其新特征，避免重复计算

        for i in range(len(edge_ids)):
            u, v = src_nodes[i].item(), dst_nodes[i].item()
            
            # --- 1. 结构扰动 (Edge Rewiring) ---
            # 将目标节点 v 重定向到一个遥远社区的节点 v'
            original_community_v = self.communities[v].item()
            target_community = original_community_v
            if self.num_communities > 1:
                while target_community == original_community_v:
                    target_community = np.random.randint(0, self.num_communities)
            
            nodes_in_target_community = (self.communities == target_community).nonzero(as_tuple=True)[0]
            
            # 排除u自身，避免创建自环
            nodes_in_target_community = nodes_in_target_community[nodes_in_target_community != u]
            if len(nodes_in_target_community) == 0:
                continue
            
            v_prime = nodes_in_target_community[torch.randperm(len(nodes_in_target_community))[:1]].item()
            
            # 新的异常边是 (u, v')
            new_edges_list.append((u, v_prime))
            
            # --- 2. 特征扰动 (Feature Disguise) ---
            # 我们扰动源节点 u 的特征，让它看起来更像新的目标 v'
            feat_u = self.features[u]
            feat_v_prime = self.features[v_prime]
            
            perturbed_feat_u = (1 - feature_mix_ratio) * feat_u + feature_mix_ratio * feat_v_prime
            perturbed_node_map[u] = perturbed_feat_u

        if not new_edges_list:
            return None, None, None

        # --- 3. 构建返回结果 ---
        perturbed_node_ids = torch.tensor(list(perturbed_node_map.keys()), device=device, dtype=torch.long)
        perturbed_node_features = torch.stack(list(perturbed_node_map.values()))
        
        return new_edges_list, perturbed_node_ids, perturbed_node_features