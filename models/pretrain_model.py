# 包含GraphMAE
from typing import Optional
from itertools import chain
from functools import partial
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl
from dgl import DGLGraph

from gnn_zoo.homogeneous_gnns import GCN, GIN, BWGNN, GraphSAGE
from utils.misc import obtain_act, obtain_norm, sce_loss
from data.anomaly_generator import AnomalyGenerator

# 边掩码。伯努利采样生成掩码索引，用于图自监督学习中的边丢弃
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

# 边丢弃
def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph
    
    E = graph.num_edges()   # 获取边的数量
    mask_rates = torch.FloatTensor(np.ones(E) * drop_rate)  # 生成掩码率
    masks = torch.bernoulli(1 - mask_rates) # 生成伯努利掩码
    edge_mask = masks.nonzero().squeeze(1)  # 获取掩码索引
    
    src, dst = graph.edges()
    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=graph.num_nodes())   # 重建子图
    ng = dgl.add_self_loop(ng)  # 添加自环保持连通性

    if return_edges:
        dsrc = src[~edge_mask]  # 获取未被掩码的源节点
        ddst = dst[~edge_mask]  # 获取未被掩码的目标节点
        return ng, (dsrc, ddst)
    return ng


def drop_nodes(g: dgl.DGLGraph, drop_rate: float = 0.2):
    """随机丢弃图中一定比例的节点"""
    if drop_rate <= 0:
        return g
    
    num_nodes = g.num_nodes()
    num_to_keep = int(num_nodes * (1 - drop_rate))
    nodes_to_keep = torch.randperm(num_nodes, device=g.device)[:num_to_keep]
    
    return dgl.node_subgraph(g, nodes_to_keep, store_ids=True)

def perturb_edges(g: dgl.DGLGraph, perturb_rate: float = 0.2):
    """随机扰动图中一定比例的边（删除或添加）"""
    if perturb_rate <= 0:
        return g
        
    num_edges_to_perturb = int(g.num_edges() * perturb_rate)
    if num_edges_to_perturb == 0:
        return g

    src, dst = g.edges()
    
    # 随机删除边
    edges_to_drop = torch.randperm(g.num_edges(), device=g.device)[:num_edges_to_perturb]
    edge_mask = torch.ones(g.num_edges(), dtype=torch.bool, device=g.device)
    edge_mask[edges_to_drop] = False
    src_kept, dst_kept = src[edge_mask], dst[edge_mask]
    
    # 随机添加边
    new_src = torch.randint(0, g.num_nodes(), (num_edges_to_perturb,), device=g.device)
    new_dst = torch.randint(0, g.num_nodes(), (num_edges_to_perturb,), device=g.device)
    
    final_src = torch.cat([src_kept, new_src])
    final_dst = torch.cat([dst_kept, new_dst])
    
    return dgl.graph((final_src, final_dst), num_nodes=g.num_nodes())

def augment_graph_view(g: dgl.DGLGraph, features: torch.Tensor, drop_node_rate: float, perturb_edge_rate: float, mask_feature_rate: float):
    """为图级别对比学习创建单个增强视图"""
    
    # 1. 边扰动
    g_aug = perturb_edges(g, perturb_edge_rate)
    
    # 2. 节点丢弃
    g_aug = drop_nodes(g_aug, drop_node_rate)
    
    # 3. 属性掩码
    # 获取子图对应的原始特征
    original_node_ids = g_aug.ndata[dgl.NID]
    features_aug = features[original_node_ids].clone()
    
    num_nodes, feat_dim = features_aug.shape
    mask = torch.rand((num_nodes, feat_dim), device=features.device) < mask_feature_rate
    features_aug[mask] = 0.0 # 简单地置为0
    
    g_aug.ndata['feature'] = features_aug
    
    return g_aug


# ======================================================================
#   Predictive SSL
# ======================================================================
# 基于掩码自编码器进行图结构预训练
class GraphMAE_PAA(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hid_dim: int,
            encoder_num_layer: int,
            decoder_num_layer: int,
            encoder_type: str = "gcn",
            decoder_type: str = "gcn",
            mask_ratio: float = 0.75,
            replace_ratio: float = 0.1,
            drop_edge_rate: float = 0.0,
            activation: str = "relu",
            norm: Optional[str] = "layernorm",
            residual: bool = True,
            loss_fn: str = "sce",
            alpha_l: float = 2.0,
            concat_hidden: bool = False,
            anomaly_generator: Optional[AnomalyGenerator] = None,
            w_recon: float = 0.5, w_contrastive: float = 1.0,
            **kwargs
         ):
        super(GraphMAE_PAA, self).__init__()
        self._mask_ratio = mask_ratio
        self._replace_ratio = replace_ratio
        self._mask_token_rate = 1.0 - self._replace_ratio
        self._drop_edge_rate = drop_edge_rate
        
        self.embed_dim = hid_dim # 预训练输出的嵌入维度
        encoder_out_dim = hid_dim
        decoder_in_dim = hid_dim

        self.anomaly_generator = anomaly_generator
        self.w_recon = w_recon
        self.w_contrastive = w_contrastive
        self.temperature = 0.1

        # 为对比损失创建一个投影头 (Projection Head)
        self.projection_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 128) # 投影到128维空间
        )

        # ======================================================================
        #   1. 编码器 (Encoder) - 支持多种GNN
        # ======================================================================
        print(f"Initializing GraphMAE_PAA Encoder: {encoder_type.upper()}")
        if encoder_type == "gcn":
            self.encoder = GCN(in_dim, hid_dim, hid_dim, encoder_num_layer, 0.0, activation, residual, norm, encoding=True)
        elif encoder_type == "gin":
            self.encoder = GIN(in_dim, hid_dim, hid_dim, encoder_num_layer, 0.0, activation, residual, norm, encoding=True)
        elif encoder_type == 'bwgnn':
            # BWGNN的输出维度是特殊的，需要单独处理
            self.encoder = BWGNN(in_dim=in_dim, num_hidden=hid_dim, encoding=True)
            # BWGNN的输出维度是 hid_dim * (d+1)，这里假设d=2
            encoder_out_dim = hid_dim * (self.encoder.d + 1)
            self.embed_dim = encoder_out_dim
            decoder_in_dim = encoder_out_dim # 解码器的输入维度也需要相应改变
        elif encoder_type == 'graphsage':
            self.encoder = GraphSAGE(
                in_dim=in_dim,
                num_hidden=hid_dim,
                out_dim=hid_dim,
                num_layers=encoder_num_layer,
                dropout=0.0, # 预训练时 dropout 通常设为 0
                activation=activation,
                residual=residual,
                norm=norm,
                encoding=True # 确保输出带有激活
            )
        else:
            raise NotImplementedError(f"Encoder type '{encoder_type}' not supported.")

        # ======================================================================
        #   2. 解码器 (Decoder) - 结构更灵活
        # ======================================================================
        print(f"Initializing GraphMAE_PAA Decoder: {decoder_type.upper()}")
        if decoder_type == "gcn":
            self.decoder = GCN(decoder_in_dim, hid_dim, in_dim, decoder_num_layer, 0.0, activation, residual, norm, encoding=False)
        elif decoder_type == "gin":
            self.decoder = GIN(decoder_in_dim, hid_dim, in_dim, decoder_num_layer, 0.0, activation, residual, norm, encoding=False)
        elif decoder_type == "bwgnn":
            self.decoder = BWGNN(in_dim=decoder_in_dim, num_hidden=hid_dim, encoding=False)
        elif decoder_type == 'graphsage':
            self.decoder = GraphSAGE(
                in_dim=decoder_in_dim,
                num_hidden=hid_dim,
                out_dim=in_dim, # 解码器输出维度是原始特征维度
                num_layers=decoder_num_layer, # 使用独立的解码器层数
                dropout=0.0,
                activation=activation,
                residual=residual,
                norm=norm,
                encoding=False # 解码器最后一层不加激活
            )
        elif decoder_type == "mlp":
            # 使用一个简单的多层MLP作为解码器
            layers = [nn.Linear(decoder_in_dim, hid_dim), nn.ReLU()]
            for _ in range(decoder_num_layer - 2):
                layers.extend([nn.Linear(hid_dim, hid_dim), nn.ReLU()])
            layers.append(nn.Linear(hid_dim, in_dim))
            self.decoder = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f"Decoder type '{decoder_type}' not supported.")

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        # 编码器到解码器的投影层，处理可能的维度不匹配（如BWGNN）
        self.encoder_to_decoder = nn.Linear(encoder_out_dim, decoder_in_dim, bias=False)
        
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            return nn.MSELoss() # 均方误差损失
        elif loss_fn == "sce":
            return partial(sce_loss, alpha=alpha_l) # 交叉熵损失，带平滑参数
        else:
            raise NotImplementedError

    def info_nce_loss(self, z1, z2):
        """标准的 InfoNCE 对比损失"""
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # 正样本在对角线上
        labels = torch.arange(z1.size(0), device=z1.device)
        
        # 对称的损失
        loss1 = F.cross_entropy(sim_matrix, labels)
        loss2 = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss1 + loss2) / 2

    def embed(self, g, x):
        """用于在下游任务中获取节点嵌入的接口"""
        # 在推理时，我们使用原始的、完整的图
        with torch.no_grad():
            if x.dtype == torch.long:
                x = x.float()
            return self.encoder(g, x)

    # ======================================================================
    #   3. 混合掩码策略 (Mixed Masking)
    # ======================================================================
    def encoding_mask_noise(self, g, x):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        
        # a. 计算需要掩码的总节点数
        num_mask_nodes = int(self._mask_ratio * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        
        out_x = x.clone()

        # 动态转换 [MASK] 令牌的数据类型以匹配输入 x
        mask_token = self.enc_mask_token.to(x.device, dtype=x.dtype)
        
        if self._replace_ratio > 0 and num_mask_nodes > 0:
            # b. 计算两种掩码策略各自的节点数
            num_token_nodes = int(num_mask_nodes * self._mask_token_rate)
            num_noise_nodes = num_mask_nodes - num_token_nodes
            
            # c. 随机选择要被 [MASK] 替换的节点和要被噪声替换的节点
            mask_perm = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[mask_perm[:num_token_nodes]]
            noise_nodes = mask_nodes[mask_perm[num_token_nodes:]]

            # 确保有节点需要被替换
            if noise_nodes.numel() > 0:
                # d. 随机选择噪声节点的替换特征
                noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:noise_nodes.size(0)]
                # e. 将噪声节点替换为随机选择的特征
                out_x[noise_nodes] = x[noise_to_be_chosen]
            
            # f. 将掩码节点替换为 [MASK] 令牌
            if token_nodes.numel() > 0:
                # 我们显式地将 self.enc_mask_token 扩展到目标形状
                # .expand() 是一个高效的操作，它不会实际复制内存
                expanded_mask_token = mask_token.expand(token_nodes.size(0), -1)
                # 替换掩码节点的特征为 [MASK] 令牌
                out_x[token_nodes] = expanded_mask_token
            
        elif num_mask_nodes > 0: # 如果不使用混合策略
            # 即使在非混合模式下，也使用显式扩展，保持代码健壮性
            expanded_mask_token = mask_token.expand(num_mask_nodes, -1)
            out_x[mask_nodes] = expanded_mask_token
            
        return out_x, mask_nodes

    def contrastive_loss(self, z_normal, z_augmented_anomaly):
        """计算正常表示和其对应的人工异常表示之间的对比损失"""
        z_normal = F.normalize(z_normal, p=2, dim=1)
        z_augmented_anomaly = F.normalize(z_augmented_anomaly, p=2, dim=1)
        
        # 使用简单的余弦相似度损失，目标是让它们不相似（即点积接近-1）
        # 或者使用更复杂的 InfoNCE 损失
        # 简单版：最大化距离，即最小化相似度
        loss = (z_normal * z_augmented_anomaly).sum(dim=1).mean()
        return loss

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        """
        场景自适应的前向传播。
        """

        # 确保输入特征 x 是浮点数类型，GNN层才能处理
        if x.dtype == torch.long:
            x_float = x.float()
        else:
            x_float = x

        # 默认执行重建损失
        loss_recon = self._forward_recon_only(g, x_float)
        
        # 如果对比损失被启用，则计算并加权
        loss_cont = torch.tensor(0.0, device=x.device)
        if self.w_contrastive > 0:
            # --- 场景一：单图 PAA ---
            if self.anomaly_generator is not None and g.batch_size == 1:
                loss_cont = self._get_paa_contrastive_loss(g, x_float)
            # --- 场景二：多图 GraphCL ---
            elif g.batch_size > 1:
                loss_cont = self._get_graphcl_contrastive_loss(g)
            
        return self.w_contrastive * loss_cont + self.w_recon * loss_recon

    def _forward_recon_only(self, g, x):
        """标准的 GraphMAE 重建损失"""
        g1 = drop_edge(g, self._drop_edge_rate)
        x1, mask_nodes = self.encoding_mask_noise(g, x)
        
        if mask_nodes.numel() == 0: return torch.tensor(0.0, device=x.device)

        if x1.dtype == torch.long:
            x1_float = x1.float()
        else:
            x1_float = x1

        rep1 = self.encoder(g1, x1)
        rep1_decoded = self.encoder_to_decoder(rep1)
        
        if isinstance(self.decoder, (GCN, GIN, BWGNN)):
            rep1_decoded[mask_nodes] = 0
        
        recon = self.decoder(g1, rep1_decoded) if not isinstance(self.decoder, nn.Sequential) else self.decoder(rep1_decoded)

        # 重建的目标是原始特征 x，其类型可能与 recon 不同
        # 在计算损失前，确保目标类型与预测类型匹配
        target_x = x[mask_nodes]
        if recon.dtype != target_x.dtype:
            target_x = target_x.to(recon.dtype)
        
        return self.criterion(recon[mask_nodes], target_x)

    def _get_paa_contrastive_loss(self, g, x):
        """单图场景下的 PAA (节点级对比 + 重建)"""
        # 1. 视图一：标准掩码
        g1 = drop_edge(g, self._drop_edge_rate)
        x1, _ = self.encoding_mask_noise(g, x)
        
        # 2. 视图二：人工异常
        num_to_aug = int(g.num_nodes() * self._mask_ratio)
        if num_to_aug == 0: return torch.tensor(0.0, device=x.device)
        nodes_to_aug = torch.randperm(g.num_nodes(), device=g.device)[:num_to_aug]
        
        aug_node_ids, perturbed_feats, perturbed_graph = self.anomaly_generator.generate_for_nodes(nodes_to_aug)
        
        if aug_node_ids is None or aug_node_ids.numel() == 0: 
            return torch.tensor(0.0, device=x.device)

        device = x.device
        aug_node_ids = aug_node_ids.to(device)
        perturbed_feats = perturbed_feats.to(device)
        perturbed_graph = perturbed_graph.to(device)

        # 3. 计算表示
        rep1 = self.encoder(g1, x1)
        
        full_perturbed_features = x.clone()
        full_perturbed_features[aug_node_ids] = perturbed_feats
        rep2_full = self.encoder(perturbed_graph, full_perturbed_features)
        
        # 4. 计算损失
        z1 = self.projection_head(rep1[aug_node_ids])
        z2 = self.projection_head(rep2_full[aug_node_ids])
        
        return self.info_nce_loss(z1, z2)

    def _get_graphcl_contrastive_loss(self, g_batched):
        """多图场景下的 GraphCL-style (图级对比)"""
        
        # 1. 解批处理，为每张图创建两个视图
        graphs = dgl.unbatch(g_batched)
        g1_list, g2_list = [], []
        # 获取原始特征，因为增强函数需要它
        original_features = g_batched.ndata['feature']

        # 确保原始特征也是浮点数
        if original_features.dtype == torch.long:
            original_features = original_features.float()
        
        # 我们需要知道每个小图的节点数，以便正确地切分特征
        nodes_per_graph = g_batched.batch_num_nodes().tolist()
        node_offsets = [0] + np.cumsum(nodes_per_graph).tolist()

        # for i, g in enumerate(graphs):
        #     # 为每张图切分出它自己的原始特征
        #     start, end = node_offsets[i], node_offsets[i+1]
        #     g_features = original_features[start:end]
            
        #     g1_list.append(augment_graph_view(g, g_features, 0.2, 0.2, 0.2))
        #     g2_list.append(augment_graph_view(g, g_features, 0.2, 0.2, 0.2))

        for i, g in enumerate(graphs):
            # =================================================================
            #   核心修复点：清洗和归一化每个图的 ndata
            # =================================================================
            # 1. 创建一个新的、干净的图，只包含结构
            cleaned_g = dgl.graph(g.edges(), num_nodes=g.num_nodes())
            
            # 2. 为新图只赋予我们需要的 'feature' 属性
            start, end = node_offsets[i], node_offsets[i+1]
            g_features = original_features[start:end]
            cleaned_g.ndata['feature'] = g_features
            
            # 3. 后续所有操作都基于这个干净的图 cleaned_g
            g1_list.append(augment_graph_view(cleaned_g, g_features, 0.2, 0.2, 0.2))
            g2_list.append(augment_graph_view(cleaned_g, g_features, 0.2, 0.2, 0.2))

        # 2. 重新批处理
        g1_batched = dgl.batch(g1_list)
        g2_batched = dgl.batch(g2_list)
        
        # 3. 计算两个视图的图级别表示
        rep1 = self.encoder(g1_batched, g1_batched.ndata['feature'])
        
        temp_feat_name = "_temp_rep1_for_pooling"
        g1_batched.ndata[temp_feat_name] = rep1
        pooled_rep1 = dgl.mean_nodes(g1_batched, feat=temp_feat_name)
        del g1_batched.ndata[temp_feat_name]
        h1 = self.projection_head(pooled_rep1)
        
        rep2 = self.encoder(g2_batched, g2_batched.ndata['feature'])
        
        temp_feat_name = "_temp_rep2_for_pooling" # 可以复用键名，但为了清晰分开写
        g2_batched.ndata[temp_feat_name] = rep2
        pooled_rep2 = dgl.mean_nodes(g2_batched, feat=temp_feat_name)
        del g2_batched.ndata[temp_feat_name]
        h2 = self.projection_head(pooled_rep2)
        
        # 4. 计算图级别对比损失
        loss_cont = self.info_nce_loss(h1, h2)
        
        return loss_cont



