# 包含AdaFreq、GNA、RHOEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import scipy.sparse as sp
from gnn_zoo.homogeneous_gnns import GCN, GIN, BWGNN

class AdaFreqFilter(nn.Module):

    def __init__(self, embed_dim: int):
        """
        Parameters
        ----------
        embed_dim : int
            The dimension of the feature channels for the channel-wise view.
        """
        super().__init__()
        # 可学习参数 k (用于跨通道视图)
        self.k_cross_channel = nn.Parameter(torch.randn(1))
        
        # 可学习参数 K (用于逐通道视图)
        # 形状为 (1, embed_dim) 以便与 (N, embed_dim) 的特征矩阵进行广播
        self.K_channel_wise = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, g: dgl.DGLGraph, H: torch.Tensor, view: str) -> torch.Tensor:
        """
        重写 forward 方法，使用 DGL 的消息传递来实现拉普拉斯滤波，
        从而完全避免手动创建和操作 PyTorch 稀疏张量。
        数学上等价于 H - k * L @ H
        """
        with g.local_scope():
            # 1. 计算 D^-1/2 * H
            degs_out = g.out_degrees().float().clamp(min=1)
            d_inv_sqrt_out = torch.pow(degs_out, -0.5).unsqueeze(-1)
            h_src_norm = H * d_inv_sqrt_out
            g.srcdata['h_norm'] = h_src_norm
            
            # 2. 消息传递，计算 A * (D^-1/2 * H)
            g.update_all(fn.copy_u('h_norm', 'm'), fn.sum('m', 'h_agg'))
            
            # 3. 计算 D^-1/2 * (A * D^-1/2 * H)
            degs_in = g.in_degrees().float().clamp(min=1)
            d_inv_sqrt_in = torch.pow(degs_in, -0.5).unsqueeze(-1)
            norm_adj_h = g.dstdata['h_agg'] * d_inv_sqrt_in
            
            # 4. 计算 L * H = (I - D^-1/2 * A * D^-1/2) * H = H - norm_adj_h
            laplacian_h = H - norm_adj_h

            if view == 'cross_channel':
                return H - self.k_cross_channel * laplacian_h
            elif view == 'channel_wise':
                # 注意：这里 K_channel_wise 作用于 laplacian_h
                return H - self.K_channel_wise * laplacian_h
            else:
                raise ValueError(f"Invalid view: {view}. Must be 'cross_channel' or 'channel_wise'.")


class GNA(nn.Module):

    def __init__(self, embed_dim: int, projection_dim: int = 128, temperature: float = 0.1):
        super().__init__()
        self.projection_head_view1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, projection_dim)
        )
        self.projection_head_view2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, projection_dim)
        )
        self.temperature = temperature

    def _sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return torch.mm(z1, z2.t())

    def forward(self, h_view1: torch.Tensor, h_view2: torch.Tensor) -> torch.Tensor:
        z_view1 = self.projection_head_view1(h_view1)
        z_view2 = self.projection_head_view2(h_view2)
        
        sim_1_to_2 = self._sim(z_view1, z_view2)
        sim_2_to_1 = sim_1_to_2.t()
        
        labels = torch.arange(z_view1.size(0), device=z_view1.device)
        
        loss_1_to_2 = F.cross_entropy(sim_1_to_2 / self.temperature, labels)
        loss_2_to_1 = F.cross_entropy(sim_2_to_1 / self.temperature, labels)
        
        return (loss_1_to_2 + loss_2_to_1) / 2

class RHOEncoder(nn.Module):

    def __init__(self, base_gnn: nn.Module, embed_dim: int, gna_projection_dim: int = 128):
        super().__init__()
        self.base_gnn = base_gnn
        self.ada_freq_filter = AdaFreqFilter(embed_dim=embed_dim)
        self.gna_module = GNA(embed_dim, projection_dim=gna_projection_dim)

    # def _get_laplacian(self, g: dgl.DGLGraph) -> torch.sparse.Tensor:
    #     """
    #     使用DGL内置函数计算归一化的拉普拉斯矩阵 L = I - D^-1/2 * A * D^-1/2。
    #     这个版本更稳定且高效。
    #     """
    #     # 1. 为图添加自环，防止节点度为0
    #     g_with_loop = dgl.add_self_loop(g)
        
    #     # 2. 获取归一化的邻接矩阵 D^-1/2 * A * D^-1/2
    #     #    DGL的gcn_norm会自动处理度的计算和归一化
    #     #    它返回的是应用在边上的权重
    #     norm_weights = dgl.norm_by_dst(g_with_loop).clamp(min=1e-12) # D^-1
    #     g_with_loop.edata['norm'] = norm_weights
    #     g_with_loop.update_all(fn.copy_e('norm', 'm'), fn.sum('m', 'c_num'))
    #     # 此时 g_with_loop.ndata['c_num'] 包含了 D^-1 * A 的对角线部分
    #     # 这是一个近似，更精确的方法是使用 dgl.laplacian_lambda_max 等
    #     # 但为了简单起见，我们直接使用 DGL 的 GCNConv 归一化方式
        
    #     # 一个更直接和正确的方法是直接使用 DGL 的归一化功能
    #     # 获取入度和出度的逆平方根
    #     degs_in = g_with_loop.in_degrees().float().clamp(min=1)
    #     degs_out = g_with_loop.out_degrees().float().clamp(min=1)
    #     d_inv_sqrt_in = torch.pow(degs_in, -0.5)
    #     d_inv_sqrt_out = torch.pow(degs_out, -0.5)
        
    #     # 将归一化系数应用到边上
    #     g_with_loop.srcdata.update({'d_inv_sqrt': d_inv_sqrt_out})
    #     g_with_loop.dstdata.update({'d_inv_sqrt': d_inv_sqrt_in})
    #     g_with_loop.apply_edges(fn.u_mul_v('d_inv_sqrt', 'd_inv_sqrt', 'norm_adj_val'))
        
    #     # 3. 构建归一化邻接矩阵的稀疏表示
    #     n = g_with_loop.num_nodes()
    #     src, dst = g_with_loop.edges()
    #     norm_adj_sparse = torch.sparse_coo_tensor(
    #         torch.stack([src, dst]),
    #         g_with_loop.edata['norm_adj_val'].squeeze(),
    #         (n, n)
    #     )
        
    #     # 4. 计算 I - norm_adj
    #     device = g.device
    #     I_sparse = torch.sparse_coo_tensor(
    #         torch.tensor([range(n), range(n)], device=device),
    #         torch.ones(n, device=device),
    #         (n, n)
    #     )
        
    #     return I_sparse - norm_adj_sparse

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor):
        device = h.device
        g = g.to(device)

        base_h = self.base_gnn(g, h)
        
        # 为 AdaFreqFilter 准备带有自环的图
        g_for_filter = dgl.add_self_loop(g)
        
        # 直接将图和特征传入滤波器
        h_ccr = self.ada_freq_filter(g_for_filter, base_h, view='cross_channel')
        h_cwr = self.ada_freq_filter(g_for_filter, base_h, view='channel_wise')

        loss_gna = None
        if self.training:
            loss_gna = self.gna_module(h_ccr, h_cwr)
            
        final_h = (h_ccr + h_cwr) / 2.0
            
        return final_h, loss_gna


    # def forward(self, g: dgl.DGLGraph, h: torch.Tensor):

    #     # 在执行任何操作前，确保图和特征在同一个设备上
    #     device = h.device
    #     g = g.to(device)

    #     # 1. 使用基础GNN提取初始嵌入
    #     # base_gnn (GCN, GIN, etc.) is expected to handle batched graphs.
    #     base_h = self.base_gnn(g, h)
        
    #     # 2. 计算拉普拉斯矩阵并进行自适应滤波
    #     L = self._get_laplacian(g) 
    #     h_ccr = self.ada_freq_filter(L, base_h, view='cross_channel')
    #     h_cwr = self.ada_freq_filter(L, base_h, view='channel_wise')

    #     # 3. 计算GNA损失 (仅在训练时)
    #     loss_gna = None
    #     if self.training:
    #         # GNA module also works on batched node features correctly.
    #         loss_gna = self.gna_module(h_ccr, h_cwr)
            
    #     # 4. 融合双视图表示作为最终输出
    #     # 简单地取平均值是一种有效且常用的融合策略
    #     final_h = (h_ccr + h_cwr) / 2.0
            
    #     return final_h, loss_gna

class RHOEncoder_NoGNA(RHOEncoder):
    """
    RHOEncoder的消融实验版本，移除了GNA自监督正则化模块。
    """
    def __init__(self, base_gnn: nn.Module, embed_dim: int, gna_projection_dim: int = 128):
        # 调用父类的构造函数来初始化 base_gnn 和 ada_freq_filter
        super().__init__(base_gnn, embed_dim, gna_projection_dim)
        
        # 覆盖父类中的 gna_module，将其设置为 None
        self.gna_module = None
        print("--- Initialized RHOEncoder_NoGNA (GNA module has been ablated) ---")

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor):
        """
        修正后的 forward 方法，与重构后的父类 RHOEncoder 保持一致。
        """
        device = h.device
        g = g.to(device)

        # 1. 使用基础GNN提取初始嵌入
        base_h = self.base_gnn(g, h)
        
        # 2. 为 AdaFreqFilter 准备带有自环的图
        g_for_filter = dgl.add_self_loop(g)
        
        # 3. 直接将图和特征传入滤波器，不再需要 L
        h_ccr = self.ada_freq_filter(g_for_filter, base_h, view='cross_channel')
        h_cwr = self.ada_freq_filter(g_for_filter, base_h, view='channel_wise')

        # 4. GNA损失始终为None
        loss_gna = None
            
        # 5. 融合双视图表示作为最终输出
        final_h = (h_ccr + h_cwr) / 2.0
            
        return final_h, loss_gna