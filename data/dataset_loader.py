import os
import dgl
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dgl.dataloading import GraphDataLoader
from data.anomaly_generator import AnomalyGenerator
import dgl.function as fn
from dgl import KHopGraph
from sklearn.preprocessing import StandardScaler

# ======================================================================
#   MRQSampler: 瑞利商子图采样器
# ======================================================================
class MRQSampler:
    """
    封装MRQ采样逻辑，用于从图中为目标节点采样一个带权重的邻居子图。
    """
    def __init__(self, strategy='star+norm'):
        self.strategy = strategy
        self.method, self.agg_feature_type = strategy.split('+') if '+' in strategy else (strategy, None)

        # 根据策略字符串，将具体的实现函数赋值给 self.sample_fn
        if self.method == 'star':
            self.get_adj_fn = self._get_star_topk_nbs
            if self.agg_feature_type == 'norm':
                self.select_topk_fn = self._select_topk_star_normft
            elif self.agg_feature_type == 'union':
                self.select_topk_fn = self._select_topk_star_unionft
            else:
                raise NotImplementedError(f"Unsupported feature type for star: {self.agg_feature_type}")
        elif self.method == 'convtree':
            if self.agg_feature_type == 'norm':
                self.get_adj_fn = self._get_convtree_topk_nbs_norm
                self.select_topk_fn = None # convtree的逻辑是统一的
            else:
                raise NotImplementedError(f"Unsupported feature type for convtree: {self.agg_feature_type}")
        elif self.method == 'khop':
            self.get_adj_fn = self._select_all_khop
            self.select_topk_fn = None
        elif self.method == 'rand':
            self.get_adj_fn = self._select_rand_khop
            self.select_topk_fn = None
        else:
            raise NotImplementedError(f"Sampling method '{self.method}' not implemented.")

    def sample(self, graph, central_node_id, khop=1):
        """
        根据初始化时设定的策略执行采样。
        返回一个DGL图，其边从邻居指向中心节点，并带有权重'pw'。
        """
        with graph.local_scope():
            # 准备归一化特征（仅部分策略需要）
            if self.agg_feature_type == 'norm':
                graph.ndata['feature_normed'] = self._prepare_normed_features(graph)

            # 准备k-hop子图上下文（仅部分策略需要）
            if self.method == 'star' or self.method == 'khop' or self.method == 'rand':
                # 这些方法需要一个预计算的k-hop图来定义邻居范围
                transform = KHopGraph(khop)
                context_graph = transform(graph)
                context_graph = dgl.to_simple(context_graph)
                context_graph = dgl.remove_self_loop(context_graph)
                # 将原始特征复制到上下文图中
                context_graph.ndata['feature'] = graph.ndata['feature']
                if 'feature_normed' in graph.ndata:
                    context_graph.ndata['feature_normed'] = graph.ndata['feature_normed']
            else: # convtree直接在原图上操作
                context_graph = graph

            # 调用具体的采样函数
            adj_list, weight_list = self.get_adj_fn(context_graph, central_node_id, khop, self.select_topk_fn)
            
            # 创建最终的带权重的采样图
            # 注意：adj_list可能是tensor或list，需要统一处理
            if isinstance(adj_list, torch.Tensor):
                src_nodes = adj_list
            else: # 可能是list of tensors
                src_nodes = torch.tensor(adj_list, dtype=torch.long)

            # 1. 确定子图包含的所有唯一节点 (邻居 + 中心节点)
            #    这些ID都是相对于原始大图的ID
            all_original_node_ids = torch.unique(torch.cat([src_nodes, torch.tensor([central_node_id])]))

            # 2. 创建一个全新的、空的DGL图，只包含这些节点。
            #    这个图没有任何边。
            num_nodes_in_subgraph = len(all_original_node_ids)
            sp_graph = dgl.graph(([], []), num_nodes=num_nodes_in_subgraph)
            
            # 3. 手动将原始ID作为 dgl.NID 特征存入这个新图。
            #    这是解决上一个 KeyError 的关键，也是解决当前问题的基础。
            sp_graph.ndata[dgl.NID] = all_original_node_ids

            orig_to_sub_id_map = {orig_id.item(): sub_id for sub_id, orig_id in enumerate(all_original_node_ids)}

            # 5. 将我们计算出的边的原始ID，转换为子图内部ID。
            sub_src_nodes = [orig_to_sub_id_map[s.item()] for s in src_nodes]
            sub_dst_node = orig_to_sub_id_map[central_node_id]

            # 6. 在这个干净的图上添加我们计算出的边。
            sp_graph.add_edges(sub_src_nodes, sub_dst_node)
            
            # 7. 现在，sp_graph.num_edges() 严格等于 len(sub_src_nodes)，
            #    也就是 len(weights)。数量匹配问题解决。
            weights = torch.tensor(weight_list, dtype=torch.float).view(-1, 1)
            sp_graph.edata['pw'] = torch.nan_to_num(weights)


            return sp_graph

    def _prepare_normed_features(self, g):
        """计算用于采样的归一化一维特征"""
        features = g.ndata['feature'].float()
        # 对于MUTAG这种one-hot特征，使用argmax
        if features.shape[1] > 1 and features.min() >= 0 and features.max() <= 1 and (features.sum(dim=1) - 1).abs().max() < 1e-6:
            # Heuristic for one-hot features
            normed_feat = features.argmax(dim=1).float()
        else:
            # 对于其他特征，进行归一化并取L2范数
            feat_min = features.min(0, keepdim=True)[0]
            feat_max = features.max(0, keepdim=True)[0]
            normed_feat = (features - feat_min) / (feat_max - feat_min + 1e-12)
            normed_feat = torch.norm(normed_feat, p=2, dim=1)
        return normed_feat

    def _select_topk_star_normft(self, graph, node_ids, central_node_id):
        h_xs = graph.ndata['feature_normed'][node_ids]
        h_x0 = graph.ndata['feature_normed'][central_node_id]
        
        up = 0
        down = torch.pow(h_x0, 2) + 1e-12
        best_rq = up / down
        
        diff_sq = torch.pow(h_x0 - h_xs, 2)
        h_xs_sq = torch.pow(h_xs, 2) + 1e-12
        greedy_scores = diff_sq / h_xs_sq
        
        sorted_indices = torch.argsort(greedy_scores, descending=True)
        
        selected_nbs = []
        for idx in sorted_indices:
            tmp_up = diff_sq[idx]
            tmp_down = h_xs_sq[idx]
            if best_rq < (up + tmp_up) / (down + tmp_down):
                up += tmp_up
                down += tmp_down
                best_rq = up / down
                selected_nbs.append(node_ids[idx].item())
            else:
                break
        return selected_nbs

    def _select_topk_star_unionft(self, graph, node_ids, central_node_id):
        h_xs = graph.ndata['feature'][node_ids]
        h_x0 = graph.ndata['feature'][central_node_id]
        
        selected_nbs_set = set()
        for feature_id in range(h_xs.shape[1]):
            h_xs_dim = h_xs[:, feature_id]
            h_x0_dim = h_x0[feature_id]
            
            up = 0
            down = torch.pow(h_x0_dim, 2) + 1e-12
            best_rq = up / down
            
            diff_sq = torch.pow(h_x0_dim - h_xs_dim, 2)
            h_xs_sq = torch.pow(h_xs_dim, 2) + 1e-12
            greedy_scores = diff_sq / h_xs_sq
            
            sorted_indices = torch.argsort(greedy_scores, descending=True)
            
            for idx in sorted_indices:
                tmp_up = diff_sq[idx]
                tmp_down = h_xs_sq[idx]
                if best_rq < (up + tmp_up) / (down + tmp_down):
                    up += tmp_up
                    down += tmp_down
                    best_rq = up / down
                    selected_nbs_set.add(node_ids[idx].item())
        return list(selected_nbs_set)

    def _get_star_topk_nbs(self, context_graph, central_node_id, khop, select_topk_fn):
        # context_graph 已经是 k-hop 图
        neighbors = torch.unique(torch.cat(context_graph.in_edges(central_node_id, form='uv')[0:1] + context_graph.out_edges(central_node_id, form='uv')[1:2]))

        if neighbors.numel() == 0:
            return [central_node_id], [1.0]

        nbs = select_topk_fn(context_graph, neighbors, central_node_id)
        nbs.append(central_node_id)
        
        weights = [0.5 / (len(nbs) -1 + 1e-12)] * (len(nbs) - 1) + [0.5]
        return nbs, weights


    def _select_all_khop(self, context_graph, central_node_id, khop, select_topk_fn):
        neighbors = torch.unique(torch.cat(context_graph.in_edges(central_node_id, form='uv')[0:1] + context_graph.out_edges(central_node_id, form='uv')[1:2]))
        if neighbors.numel() == 0:
            return [central_node_id], [1.0]
        
        weights = [1.0 / (neighbors.shape[0] + 1)] * neighbors.shape[0] + [1.0 / (neighbors.shape[0] + 1)]
        return torch.cat([neighbors, torch.tensor([central_node_id])]), weights

    def _select_rand_khop(self, context_graph, central_node_id, khop, select_topk_fn, k=100):
        neighbors = torch.unique(torch.cat(context_graph.in_edges(central_node_id, form='uv')[0:1] + context_graph.out_edges(central_node_id, form='uv')[1:2]))
        if neighbors.numel() == 0:
            return [central_node_id], [1.0]
        
        if neighbors.shape[0] > k:
            idx = torch.randperm(neighbors.shape[0])
            neighbors = neighbors[idx[:k]]
        
        weights = [1.0 / (neighbors.shape[0] + 1)] * neighbors.shape[0] + [1.0 / (neighbors.shape[0] + 1)]
        return torch.cat([neighbors, torch.tensor([central_node_id])]), weights

    # 树型卷积采样
    def _get_convtree_topk_nbs_norm(self, graph, central_node_id, khop, select_topk_fn):
        """
        基于2-hop卷积树的瑞利商最大化采样。
        返回邻居ID列表和对应的权重列表。
        """
        # 确保khop为2，因为此算法是为2-hop设计的
        assert khop == 2, "Convolutional Tree Sampler is designed for khop=2."

        xi = central_node_id
        
        # 获取一跳邻居
        # 使用DGL的in_edges和out_edges来获取所有邻居
        neighbors_in, _ = graph.in_edges(xi)
        _, neighbors_out = graph.out_edges(xi)
        nbs_xi = torch.unique(torch.cat([neighbors_in, neighbors_out]))
        
        if nbs_xi.numel() == 0:
            # 没有邻居，只返回自身
            return [xi], [1.0]
            
        xf = graph.ndata['feature_normed']
        
        # Pij: 存储一跳邻居xj的权重
        # Pik: 存储二跳邻居xk的权重
        Pij, Pik = {}, {}
        # Pij_tmp: 临时存储xj的初始权重，在确定最终选择的一跳邻居数量后再进行归一化
        Pij_tmp = {}
        # Smaxj_list: 存储每个一跳邻居xj及其子树能达到的最大瑞利商信息 (xj_id, a_j_max, b_j_max)
        Smaxj_list = []

        # --- 阶段一：为每个一跳邻居xj，计算其子树的最优瑞利商 ---
        for xj_tensor in nbs_xi:
            xj = xj_tensor.item()
            
            # 计算(xi, xj)这条边的瑞利商贡献
            aj = torch.pow(xf[xj] - xf[xi], 2)
            bj = torch.pow(xf[xj], 2) + 1e-12
            Smaxj = aj / bj
            
            # 获取xj的邻居（即xi的二跳邻居）
            xj_neighbors_in, _ = graph.in_edges(xj)
            _, xj_neighbors_out = graph.out_edges(xj)
            nbs_xj = torch.unique(torch.cat([xj_neighbors_in, xj_neighbors_out]))
            # 排除父节点xi
            nbs_xj = nbs_xj[nbs_xj != xi]

            if nbs_xj.numel() == 0:
                # 如果xj没有其他邻居，它自己就是一个独立的子树
                Pij_tmp[xj] = 0.5 # 初始权重，假设xi和xj各占一半
            else:
                # 如果xj有其他邻居，它的初始权重降低，因为需要和它的子节点分享
                Pij_tmp[xj] = 0.25 # 假设xi, xj, xj的子节点群，三者权重相关
                
                # 计算所有二跳邻居xk对xj的瑞利商贡献
                ss = []
                for xk_tensor in nbs_xj:
                    xk = xk_tensor.item()
                    ak = torch.pow(xf[xk] - xf[xj], 2)
                    bk = torch.pow(xf[xk], 2) + 1e-12
                    ss.append({'id': xk, 'a': ak, 'b': bk, 'rq': ak / bk})
                
                # 按瑞利商贡献从大到小排序
                ss.sort(key=lambda x: -x['rq'])
                
                # 贪心选择二跳邻居
                selected_sons = []
                for son in ss:
                    # 判断加入这个二跳邻居是否能提升xj子树的瑞利商
                    if son['rq'] > Smaxj:
                        aj += son['a']
                        bj += son['b']
                        Smaxj = aj / bj
                        selected_sons.append(son)
                    else:
                        break
                
                # 如果选择了二跳邻居，更新它们的权重
                if selected_sons:
                    weight_per_son = 0.25 / len(selected_sons)
                    for son in selected_sons:
                        Pik[son['id']] = Pik.get(son['id'], 0) + weight_per_son
                else:
                    # 如果没有选择任何二跳邻居，xj的权重恢复为0.5
                    Pij_tmp[xj] = 0.5
            
            Smaxj_list.append({'id': xj, 'a': aj, 'b': bj, 'rq': Smaxj})

        # --- 阶段二：在所有一跳邻居中，选择最优的组合 ---
        # 按每个子树能达到的最大瑞利商从大到小排序
        Smaxj_list.sort(key=lambda x: -x['rq'])
        
        # 至少选择瑞利商最大的那个一跳邻居
        best_xj_info = Smaxj_list[0]
        ai = best_xj_info['a']
        bi = best_xj_info['b']
        RQ_max = ai / bi
        
        # 将被选中的一跳邻居加入Pij
        Pij[best_xj_info['id']] = Pij_tmp[best_xj_info['id']]
        
        # 贪心选择其他一跳邻居
        for xj_info in Smaxj_list[1:]:
            # 判断加入这个一跳邻居子树是否能提升总体的瑞利商
            if xj_info['rq'] > RQ_max:
                ai += xj_info['a']
                bi += xj_info['b']
                RQ_max = ai / bi
                Pij[xj_info['id']] = Pij_tmp[xj_info['id']]
            else:
                break
        
        # --- 阶段三：归一化权重 ---
        # 归一化一跳邻居的权重
        num_hop1_selected = len(Pij)
        for xj_id in Pij:
            Pij[xj_id] /= num_hop1_selected
            
        # 中心节点权重
        Pij[xi] = 0.5
        
        # 合并一跳和二跳邻居的权重
        P_final = Pij.copy()
        for xk_id, weight in Pik.items():
            P_final[xk_id] = P_final.get(xk_id, 0) + weight
            
        # 提取最终的邻居列表和权重列表
        adj_list = list(P_final.keys())
        weight_list = list(P_final.values())
        
        return adj_list, weight_list


# ======================================================================
#   UniGADDataset: 数据集加载与处理类
# ======================================================================
class UniGADDataset(torch.utils.data.Dataset):
    def __init__(self, name, data_dir, sp_type='star+norm', debug_num=-1):
        self.name = name
        self.data_dir = data_dir
        self.sp_type = sp_type
        self.is_single_graph = False
        
        # 定义单图数据集的简称列表 (除了这些都是多图)
        SINGLEGRAPH_NAMES = [
            'reddit', 'weibo', 'amazon', 'yelp', 
            'tfinance', 'tolokers', 'questions', 'amazon_no_purify', 'yelp_no_purify'
        ]

        if name in SINGLEGRAPH_NAMES:
            self.is_single_graph = True
            full_path = os.path.join(data_dir, 'edge_labels', f"{name}-els")
            print(f"Loading single-graph dataset from: {full_path}")
            self.graph_list, _ = dgl.load_graphs(full_path)
            self.graph_labels = torch.tensor([])
        else: # Multi-graph
            self.is_single_graph = False
            full_path = os.path.join(data_dir, 'unified', name)
            print(f"Loading multi-graph dataset from: {full_path}")
            self.graph_list, labels_dict = dgl.load_graphs(full_path)
            self.graph_labels = labels_dict.get('glabel', torch.tensor([]))

        if not self.graph_list:
            raise FileNotFoundError(f"No graphs loaded from path: {full_path}. Please check the path and data.")

        self._normalize_features()

        # 确定标签存在情况
        self.labels_have = ''
        if 'node_label' in self.graph_list[0].ndata and self.graph_list[0].ndata['node_label'].numel() > 0: self.labels_have += 'n'
        if self.is_single_graph and 'edge_label' in self.graph_list[0].edata and self.graph_list[0].edata['edge_label'].numel() > 0: self.labels_have += 'e'
        if self.graph_labels.numel() > 0: self.labels_have += 'g'

        if debug_num > 0:
            self.graph_list = self.graph_list[:debug_num]
            if self.graph_labels.numel() > 0:
                self.graph_labels = self.graph_labels[:debug_num]

        self.in_dim = self.graph_list[0].ndata['feature'].shape[1]
        
        self.sampler = MRQSampler(sp_type) if sp_type else None
        
        self.split_masks = {}
        self.anomaly_generator = None
        if self.is_single_graph:
            # AnomalyGenerator 只在单图场景下有意义，因为它需要一个固定的图来进行社区发现
            # 将原始图移动到CPU上进行社区发现，避免占用GPU内存
            # to_cpu() 确保了即使图在GPU上也能正常工作
            graph_for_community_detection = self.graph_list[0].to('cpu')
            self.anomaly_generator = AnomalyGenerator(graph_for_community_detection)


    def _normalize_features(self):
        """
        内部辅助函数，检查并标准化图的节点特征。
        对大规模多图数据集进行了优化。
        """
        print("Checking and normalizing features...")
        # 检查第一个图的特征来决定是否需要标准化
        if not self.graph_list or 'feature' not in self.graph_list[0].ndata:
            print("No features found to normalize.")
            return
            
        features = self.graph_list[0].ndata['feature']
        needs_normalization = features.dtype == torch.long or \
                              (features.dtype == torch.float and features.abs().max() > 10)

        if not needs_normalization:
            print("Features are already in a reasonable range. Skipping normalization.")
            return

        print(f"Normalizing features for dataset '{self.name}'...")
        if self.is_single_graph:
            g = self.graph_list[0]
            features = g.ndata['feature'].float()
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features.numpy())
            g.ndata['feature'] = torch.from_numpy(scaled_features).float()
        else: # Multi-graph
            # 对于大规模多图，一次性拼接所有特征可能会爆内存。
            # 我们采用更稳健的在线均值和方差计算方法。
            scaler = StandardScaler()
            
            # 分块处理以避免内存溢出
            # 先用一部分数据 fit scaler
            print("Fitting StandardScaler on a subset of features...")
            subset_features = torch.cat([g.ndata['feature'] for g in self.graph_list[:1000]], dim=0).float()
            scaler.fit(subset_features.numpy())
            
            # 然后逐个图进行 transform
            print("Transforming features for all graphs...")
            for g in tqdm(self.graph_list, desc="Normalizing graph features"):
                if 'feature' in g.ndata and g.num_nodes() > 0:
                    features = g.ndata['feature'].float()
                    scaled_features = scaler.transform(features.numpy())
                    g.ndata['feature'] = torch.from_numpy(scaled_features).float()

    def prepare_split(self, trial_id=0, seed=42):
        """为指定的实验次数(trial_id)准备数据划分"""
        if trial_id in self.split_masks:
            return self.split_masks[trial_id]

        print(f"Preparing split for trial {trial_id} with seed {seed}...")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 默认划分比例
        train_ratio, val_ratio = 0.4, 0.2 
        # 根据 UniGAD 论文 Table 1 进行定制
        if self.name in ['amazon', 'yelp', 'amazon_no_purify', 'yelp_no_purify']:
            train_ratio, val_ratio = 0.7, 0.1 # 70% 训练, 10% 验证, 20% 测试
        elif self.name in ['tolokers', 'questions', 'tfinance']:
            train_ratio, val_ratio = 0.5, 0.25 # 50% 训练, 25% 验证, 25% 测试
        elif self.name in ['reddit', 'weibo']:
            train_ratio, val_ratio = 0.4, 0.2 # 40% 训练, 20% 验证, 40% 测试
        # 对于多图数据集，也可以添加类似逻辑
        elif self.name in ['bm/dgl/bm_mn_dgl', 'bm/dgl/bm_ms_dgl', 'bm/dgl/bm_mt_dgl', 'mutag/dgl/mutag', 'TGroup']:
             train_ratio, val_ratio = 0.4, 0.2 # UniGAD 也是 40%
        elif self.name in ['mnist_processed/mnist0_strict', 'mnist_processed/mnist1_strict']:
             train_ratio, val_ratio = 0.1, 0.1 # UniGAD 是 10%

        print(f"  - Using split ratio for '{self.name}': train={train_ratio*100}%, val={val_ratio*100}%")

        if self.is_single_graph:
            g = self.graph_list[0]
    
            # --- 节点划分 ---
            idx_train_n, idx_val_n, idx_test_n = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
            if 'n' in self.labels_have:
                num_nodes = g.num_nodes()
                node_indices = np.arange(num_nodes)
                node_labels = g.ndata['node_label']
                
                # 健壮性检查
                stratify_n = None
                unique_labels_n, counts_n = np.unique(node_labels.numpy(), return_counts=True)
                if np.all(counts_n >= 2):
                    stratify_n = node_labels
                else:
                    print(f"Warning: Cannot stratify node split for '{self.name}'. Falling back to random splitting.")

                # 第一次划分
                idx_train_n, idx_rest_n, _, _ = train_test_split(node_indices, node_labels, stratify=stratify_n, train_size=train_ratio, random_state=seed)
                
                # 第二次划分
                if val_ratio > 0 and len(idx_rest_n) > 1:
                    remaining_ratio = 1.0 - train_ratio
                    val_size_in_rest = val_ratio / remaining_ratio if remaining_ratio > 0 else 0
                    if 0 < val_size_in_rest < 1:
                        stratify_rest_n = node_labels[idx_rest_n] if stratify_n is not None else None
                        # ... (可以添加更精细的二次分层检查) ...
                        idx_val_n, idx_test_n, _, _ = train_test_split(idx_rest_n, stratify_rest_n, train_size=val_size_in_rest, random_state=seed)
                    else:
                        idx_test_n = idx_rest_n # 剩余的全是测试集
                else:
                    idx_test_n = idx_rest_n

            # --- 边划分 ---
            idx_train_e, idx_val_e, idx_test_e = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
            if 'e' in self.labels_have:
                num_edges = g.num_edges()
                edge_indices = np.arange(num_edges)
                edge_labels = g.edata['edge_label']
                
                # 健壮性检查
                stratify_e = None
                unique_labels_e, counts_e = np.unique(edge_labels.numpy(), return_counts=True)
                if np.all(counts_e >= 2):
                    stratify_e = edge_labels
                else:
                    print(f"Warning: Cannot stratify edge split for '{self.name}'. Falling back to random splitting.")

                # 第一次划分
                idx_train_e, idx_rest_e, _, _ = train_test_split(edge_indices, edge_labels, stratify=stratify_e, train_size=train_ratio, random_state=seed)
                
                # 第二次划分
                if val_ratio > 0 and len(idx_rest_e) > 1:
                    remaining_ratio = 1.0 - train_ratio
                    val_size_in_rest = val_ratio / remaining_ratio if remaining_ratio > 0 else 0
                    if 0 < val_size_in_rest < 1:
                        stratify_rest_e = edge_labels[idx_rest_e] if stratify_e is not None else None
                        # ... (可以添加更精细的二次分层检查) ...
                        idx_val_e, idx_test_e, _, _ = train_test_split(idx_rest_e, stratify_rest_e, train_size=val_size_in_rest, random_state=seed)
                    else:
                        idx_test_e = idx_rest_e
                else:
                    idx_test_e = idx_rest_e



            # 将划分好的索引存入 masks
            self.split_masks[trial_id] = {
                'train': {'n': torch.from_numpy(idx_train_n), 'e': torch.from_numpy(idx_train_e), 'g': torch.tensor([])},
                'val':   {'n': torch.from_numpy(idx_val_n),   'e': torch.from_numpy(idx_val_e),   'g': torch.tensor([])},
                'test':  {'n': torch.from_numpy(idx_test_n),  'e': torch.from_numpy(idx_test_e),  'g': torch.tensor([])}
            }

            
        else:
            num_graphs = len(self.graph_list)
            graph_indices = np.arange(num_graphs)
            
            # 健壮性检查：只有当每个类别样本数都足够时才进行分层采样
            stratify_labels = None
            if self.graph_labels.numel() > 0:
                unique_labels, counts = np.unique(self.graph_labels.numpy(), return_counts=True)
                # 至少需要 train/val/test 各一个样本，所以一个类别至少需要3个？
                # 一个更安全的检查是，每个类别至少有2个样本，才能进行第一次划分
                if np.all(counts >= 2):
                    stratify_labels = self.graph_labels
                else:
                    print(f"Warning: Cannot stratify multi-graph split for '{self.name}'. Falling back to random splitting.")

            # 第一次划分：训练集 vs. (验证集 + 测试集)
            idx_train_g, idx_rest_g, _, _ = train_test_split(
                graph_indices, self.graph_labels, stratify=stratify_labels, 
                train_size=train_ratio, random_state=seed
            )
            
            # 第二次划分：从剩余部分中划分出验证集和测试集
            idx_val_g = np.array([], dtype=int)
            idx_test_g = idx_rest_g # 默认剩余部分全是测试集

            if val_ratio > 0 and len(idx_rest_g) > 1:
                # 计算验证集在剩余部分中的比例
                remaining_ratio = 1.0 - train_ratio
                val_size_in_rest = val_ratio / remaining_ratio if remaining_ratio > 0 else 0
                
                # 如果计算出的比例有效
                if 0 < val_size_in_rest < 1:
                    stratify_rest = self.graph_labels[idx_rest_g] if stratify_labels is not None else None
                    if stratify_rest is not None:
                        # 检查剩余部分是否还能分层
                        unique_labels_rest, counts_rest = np.unique(stratify_rest.numpy(), return_counts=True)
                        if not np.all(counts_rest >= 1): # 至少要保证每个类别在val和test中各出现一次
                            stratify_rest = None
                    
                    idx_val_g, idx_test_g, _, _ = train_test_split(
                        idx_rest_g, stratify_rest,
                        train_size=val_size_in_rest,
                        random_state=seed
                    )

            self.split_masks[trial_id] = {
                'train': {'g': torch.from_numpy(idx_train_g)},
                'val':   {'g': torch.from_numpy(idx_val_g)},
                'test':  {'g': torch.from_numpy(idx_test_g)}
            }
        return self.split_masks[trial_id]

    def get_subset(self, split_name, trial_id=0):
        """获取指定划分的子数据集"""
        if trial_id not in self.split_masks:
            self.prepare_split(trial_id)
        
        masks = self.split_masks[trial_id][split_name]
        return Subset(self, masks)

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        """返回一个图及其所有标签"""
        g = self.graph_list[idx]
        labels = {
            'n': g.ndata.get('node_label', torch.tensor([])),
            'e': g.edata.get('edge_label', torch.tensor([])),
            'g': self.graph_labels[idx] if self.graph_labels.numel() > 0 else torch.tensor([])
        }
        # 注意：MRQSampler的采样（khop_graph）是在模型层面或collate_fn中处理，
        # 因为它依赖于具体的节点/边目标，而不是整个图。
        # 这里我们只返回原始图。
        return g, labels

class Subset(torch.utils.data.Dataset):
    """用于表示数据集子集的辅助类"""
    def __init__(self, dataset, masks):
        self.dataset = dataset
        self.masks = masks
        
        if self.dataset.is_single_graph:
            # 对于单图，我们关心的是节点/边的索引
            self.node_indices = masks.get('n', torch.tensor([])).tolist()
            self.edge_indices = masks.get('e', torch.tensor([])).tolist()
            # 长度是所有任务样本的总和
            self.length = len(self.node_indices) + len(self.edge_indices)
        else:
            # 对于多图，我们关心的是图的索引
            self.graph_indices = masks.get('g', torch.tensor([])).tolist()
            self.length = len(self.graph_indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.dataset.is_single_graph:
            # 返回 (原始图, 任务类型, 目标ID, 标签)
            g = self.dataset.graph_list[0]
            if idx < len(self.node_indices):
                target_id = self.node_indices[idx]
                label = g.ndata['node_label'][target_id]
                return g, 'n', target_id, label
            else:
                idx -= len(self.node_indices)
                target_id = self.edge_indices[idx]
                label = g.edata['edge_label'][target_id]
                return g, 'e', target_id, label
        else:
            # 返回 (图, 标签字典)
            graph_idx = self.graph_indices[idx]
            return self.dataset[graph_idx]


# ======================================================================
#   Collate Function: 批处理函数
# ======================================================================
def collate_fn_unify(samples, 
                     sampler=None,
                     anomaly_generator=None, 
                     aug_ratio=0.5, 
                     num_perturb_edges=5, 
                     feature_mix_ratio=0.5, use_node_aug=False, use_edge_aug=False):
    """
    将样本列表批处理成模型所需的格式。
    这个函数非常关键，它连接了数据和模型。
    """
    # samples 的格式取决于数据集类型
    if isinstance(samples[0][0], dgl.DGLGraph) and isinstance(samples[0][1], dict):
        # 多图数据集: samples is a list of (graph, labels_dict)
        graphs, labels_list = map(list, zip(*samples))
        # batched_graph = dgl.batch(graphs)

        cleaned_graphs = []
        for g in graphs:
            # 1. 创建一个只包含结构的新图
            cleaned_g = dgl.graph(g.edges(), num_nodes=g.num_nodes())
            # 2. 只迁移我们需要的 'feature' 和 'node_label'
            if 'feature' in g.ndata:
                cleaned_g.ndata['feature'] = g.ndata['feature']
            if 'node_label' in g.ndata:
                cleaned_g.ndata['node_label'] = g.ndata['node_label']
            cleaned_graphs.append(cleaned_g)

        # 使用清洗后的图列表进行批处理
        batched_graph = dgl.batch(cleaned_graphs)
        
        # 准备模型输入
        # 预计算的嵌入将在训练器中完成，这里只准备图和节点标签
        g_info = {'n': batched_graph, 'g': batched_graph}
        
        # 准备标签
        batched_labels = {}
        if labels_list[0]['n'].numel() > 0:
            batched_labels['n'] = torch.cat([d['n'] for d in labels_list])

        if labels_list[0]['g'].numel() > 0:
            batched_labels['g'] = torch.stack([d['g'] for d in labels_list])
            
        return batched_graph, batched_labels, g_info

    elif isinstance(samples[0][0], dgl.DGLGraph) and isinstance(samples[0][1], str):
        tasks = {'n': [], 'e': []}
        for g, task_type, target_id, label in samples:
            tasks[task_type].append((target_id, label))
        
        original_graph = samples[0][0]

        # 1. 正常地处理原始批次中的样本
        tasks = {'n': [], 'e': []}
        for _, task_type, target_id, label in samples:
            tasks[task_type].append({'id': target_id, 'label': label})

        batched_labels = {}
        subgraphs_dict = {}

        # 1. 先从原始大图中获取所有节点的特征
        original_features = original_graph.ndata['feature']

        # 处理节点任务
        if tasks['n']:
            # 现在 s 是一个字典，s['id'] 和 s['label'] 可以正常工作
            target_ids_n = [s['id'] for s in tasks['n']]
            labels_n = [s['label'] for s in tasks['n']]
            batched_labels['n'] = torch.tensor(labels_n, dtype=torch.long)
            
            subgraph_list_n = []
            if sampler:
                for tid in target_ids_n:
                    # MRQSampler 返回的子图只包含结构
                    subgraph = sampler.sample(original_graph, tid)
                    # 手动将原始特征复制到子图中
                    # 子图的节点ID是原始图ID的子集
                    original_node_ids = subgraph.ndata[dgl.NID]
                    subgraph.ndata['feature'] = original_features[original_node_ids]
                    subgraph_list_n.append(subgraph)
            else:   # 如果没有采样器，使用简单的k-hop作为回退
                for tid in target_ids_n:
                    subgraph = dgl.khop_graph(original_graph, tid, k=1)
                    subgraph.ndata['feature'] = original_features[subgraph.nodes()]
                    subgraph_list_n.append(subgraph)
            
            if subgraph_list_n:
                subgraphs_dict['n'] = dgl.batch(subgraph_list_n)

        # 处理边任务
        if tasks['e']:
            # 现在 s 是一个字典，s['id'] 和 s['label'] 可以正常工作
            target_ids_e = [s['id'] for s in tasks['e']]
            labels_e = [s['label'] for s in tasks['e']]
            batched_labels['e'] = torch.tensor(labels_e, dtype=torch.long)

            edge_ids_tensor = torch.tensor(target_ids_e, dtype=original_graph.idtype)
            
            # 边任务的“子图”是其端点节点构成的图
            # 我们不使用sampler，而是直接提取端点
            src_nodes, dst_nodes = original_graph.find_edges(edge_ids_tensor)
            
            # 为每条边创建一个只包含其两个端点的子图
            # 注意：这在批处理时效率较低，一个更好的方法是创建一个包含所有相关端点的大图
            # 但为了逻辑清晰，我们先用这个方法
            edge_subgraph_list = []
            for i in range(len(target_ids_e)):
                nodes_for_edge = torch.unique(torch.tensor([src_nodes[i], dst_nodes[i]], dtype=original_graph.idtype))
                subgraph = dgl.node_subgraph(original_graph, nodes_for_edge, store_ids=True)
                # 同样需要复制特征
                # dgl.node_subgraph 会自动保留原始节点ID在 .ndata[dgl.NID] 中
                original_node_ids = subgraph.ndata[dgl.NID]
                subgraph.ndata['feature'] = original_features[original_node_ids]
                # node_subgraph 会保留原始节点ID
                edge_subgraph_list.append(subgraph)
            
            if edge_subgraph_list:
                subgraphs_dict['e'] = dgl.batch(edge_subgraph_list)
                subgraphs_dict['e'].ndata['feature'] = original_features[subgraphs_dict['e'].ndata[dgl.NID]]

            
        # 2. 人工异常生成部分
        if anomaly_generator and aug_ratio > 0:
            # 节点任务生成人工异常节点
            if use_node_aug and 'n' in tasks and tasks['n']:
                # 2. 从当前批次的【正常】节点中，选择一部分作为扰动目标
                normal_node_samples = [s for s in tasks['n'] if s['label'] == 0]
                if normal_node_samples:
                    num_to_generate = int(len(normal_node_samples) * aug_ratio)
                    if num_to_generate > 0:
                        # 从正常样本中随机选择，不放回
                        nodes_to_perturb_info = np.random.choice(normal_node_samples, num_to_generate, replace=False)
                        nodes_to_perturb_ids = torch.tensor([s['id'] for s in nodes_to_perturb_info], dtype=torch.int32)
                        
                        # 3. 调用生成器，得到扰动后的特征和【扰动后的全图结构】以及扰动的节点ID
                        aug_node_ids, perturbed_features, perturbed_graph = anomaly_generator.generate_for_nodes(
                            nodes_to_perturb_ids, 
                            num_perturb_edges=num_perturb_edges, 
                            feature_mix_ratio=feature_mix_ratio
                        )
                        
                        if aug_node_ids is not None:
                            aug_subgraph_list = []
                            for aug_node_id in aug_node_ids: # 使用返回的 aug_node_ids
                                subgraph = sampler.sample(perturbed_graph, aug_node_id.item())
                                aug_subgraph_list.append(subgraph)
                            
                            aug_batched_subgraph = dgl.batch(aug_subgraph_list)
                            
                            all_subgraph_node_ids = aug_batched_subgraph.ndata[dgl.NID]
                            features_for_subgraph_nodes = original_features[all_subgraph_node_ids]
                            
                            perturbed_feature_map = {
                                node_id.item(): feature 
                                for node_id, feature in zip(aug_node_ids, perturbed_features) # 使用 aug_node_ids
                            }
                            
                            # 遍历批处理子图中的每个节点
                            for i, orig_id in enumerate(all_subgraph_node_ids):
                                # 如果这个节点是我们扰动的中心节点之一
                                if orig_id.item() in perturbed_feature_map:
                                    # 用扰动后的特征替换它的原始特征
                                    features_for_subgraph_nodes[i] = perturbed_feature_map[orig_id.item()]

                            # 9. 将构建好的、维度匹配的完整特征张量赋予批处理子图
                            aug_batched_subgraph.ndata['feature'] = features_for_subgraph_nodes
                            
                            # 10. 合并到最终的输出中
                            if 'n' in subgraphs_dict:
                                # 如果原始批次中就有节点任务
                                subgraphs_dict['n'] = dgl.batch([subgraphs_dict['n'], aug_batched_subgraph])
                                aug_labels = torch.ones(num_to_generate, dtype=torch.long)
                                batched_labels['n'] = torch.cat([batched_labels['n'], aug_labels])
                            else:
                                # 如果原始批次中没有节点任务（极小概率），直接赋值
                                subgraphs_dict['n'] = aug_batched_subgraph
                                batched_labels['n'] = torch.ones(num_to_generate, dtype=torch.long)
            # 边任务生成人工异常边
            if use_edge_aug and 'e' in tasks and tasks['e']:
                normal_edge_samples = [s for s in tasks['e'] if s['label'] == 0]
                if normal_edge_samples:
                    num_to_generate = int(len(normal_edge_samples) * aug_ratio)
                    if num_to_generate > 0:
                        edges_to_perturb_info = np.random.choice(normal_edge_samples, num_to_generate, replace=False)
                        edges_to_perturb_ids = torch.tensor([s['id'] for s in edges_to_perturb_info], dtype=original_graph.idtype)
                        
                        # a. 调用新的 generate_for_edges 方法
                        new_edges, perturbed_ids, perturbed_feats = anomaly_generator.generate_for_edges(
                            edges_to_perturb_ids, feature_mix_ratio
                        )
                        
                        if new_edges is not None:
                            # b. 为新生成的异常边构建上下文子图
                            aug_edge_subgraph_list = []
                            for u, v_prime in new_edges:
                                # 子图包含新的端点 u 和 v'
                                nodes_for_edge = torch.unique(torch.tensor([u, v_prime], dtype=original_graph.idtype))
                                # 使用原始图结构来采样邻居，模拟“异常关系出现在正常环境中”
                                subgraph = dgl.node_subgraph(original_graph, nodes_for_edge, store_ids=True)
                                aug_edge_subgraph_list.append(subgraph)
                            
                            aug_batched_subgraph = dgl.batch(aug_edge_subgraph_list)
                            
                            # c. 为这个批处理子图构建正确的、混合的节点特征
                            all_subgraph_node_ids = aug_batched_subgraph.ndata[dgl.NID]
                            features_for_subgraph_nodes = original_features[all_subgraph_node_ids]
                            
                            perturbed_feature_map = {nid.item(): feat for nid, feat in zip(perturbed_ids, perturbed_feats)}
                            
                            for i, orig_id in enumerate(all_subgraph_node_ids):
                                if orig_id.item() in perturbed_feature_map:
                                    features_for_subgraph_nodes[i] = perturbed_feature_map[orig_id.item()]
                            
                            aug_batched_subgraph.ndata['feature'] = features_for_subgraph_nodes
                            
                            # d. 合并到最终的输出中
                            if 'e' in subgraphs_dict:
                                subgraphs_dict['e'] = dgl.batch([subgraphs_dict['e'], aug_batched_subgraph])
                                aug_labels = torch.ones(len(new_edges), dtype=torch.long)
                                batched_labels['e'] = torch.cat([batched_labels['e'], aug_labels])
                            else: # 如果原始批次没有边任务
                                subgraphs_dict['e'] = aug_batched_subgraph
                                batched_labels['e'] = torch.ones(len(new_edges), dtype=torch.long)
        
        
        # 对于单图，第一个返回值是原始大图，用于可能的全局上下文
        return original_graph, batched_labels, subgraphs_dict

    else:
        raise TypeError(f"Unsupported sample format: {type(samples[0][0])}, {type(samples[0][1])}")