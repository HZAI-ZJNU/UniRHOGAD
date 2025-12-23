# 包含GCN、GIN、BWGNN等
# Adapt from: https://github.com/THUDM/GraphMAE/blob/main/graphmae/models/edcoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair
import sympy
import scipy
from dgl.nn.pytorch.conv import SAGEConv
from utils.misc import obtain_act, obtain_norm

class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False
                 ):
        super(GCN, self).__init__()
        # 初始化参数
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()   # 存储GCN层
        self.activation = activation    # 激活函数
        self.dropout = dropout  # dropout率

        # 使用工厂函数将字符串转换为类
        norm_layer = obtain_norm(norm) 


        # 最后一层配置（编码器与解码器不同）
        # 如果是编码器，则最后一层使用激活函数和归一化，否则不使用（last_activation, last_residual, last_norm）
        last_activation = obtain_act(activation) if encoding else None
        last_residual = encoding and residual

        # 如果是编码器，则最后一层使用转换后的 norm_layer
        last_norm = norm_layer if encoding else None 

        
        # 构建网络层
        # 如果只有一层，则直接添加一个GraphConv层（输入到输出）
        if num_layers == 1:
            # 单层直接映射 in_dim -> out_dim
            self.gcn_layers.append(GraphConv(
                in_dim, out_dim, residual=last_residual, norm=last_norm, activation=last_activation))
        else:
            # input projection (no residual)
            # 第一层：in_dim -> num_hidden  使用残差（如果residual为True）和归一化，以及激活函数
            self.gcn_layers.append(GraphConv(
                in_dim, num_hidden, residual=residual, norm=norm_layer, activation=obtain_act(activation)))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                # 中间层：num_hidden -> num_hidden  同样使用残差、归一化和激活函数
                self.gcn_layers.append(GraphConv(
                    num_hidden, num_hidden, residual=residual, norm=norm_layer, activation=obtain_act(activation)))
            # output projection
            # 最后一层：num_hidden -> out_dim   使用last_residual（如果是编码器且residual为True）和last_norm（如果是编码器则使用norm），以及last_activation（如果是编码器则使用激活函数）
            self.gcn_layers.append(GraphConv(
                num_hidden, out_dim, residual=last_residual, activation=last_activation, norm=last_norm))

        self.norms = None   # 归一化层（代码中未使用）
        self.head = nn.Identity()   # 输出头（初始化为恒等映射）

    def forward(self, g, inputs, return_hidden=False):
        if inputs.dtype == torch.long:
            h = inputs.float()
        else:
            h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)    # dropout
            h = self.gcn_layers[l](g, h)    # 图卷积
            if self.norms is not None and l != self.num_layers - 1:
                h = self.norms[l](h)
            hidden_list.append(h)   # 记录隐藏层
        # output projection
        if self.norms is not None and len(self.norms) == self.num_layers:
            h = self.norms[-1](h)
        # 返回结果
        if return_hidden:
            return self.head(h), hidden_list    # 返回输出和所有隐藏层
        else:
            return self.head(h) # 仅返回输出

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)    # 重置分类头（用于下游任务），将head替换为一个线性层（输出维度为num_classes）


# 图卷积层
class GraphConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm=None,
                 activation=None,
                 residual=True,
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim

        self.fc = nn.Linear(in_dim, out_dim)    # 线性变换层

        # 残差连接配置
        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)    # 维度不匹配时使用线性映射
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()     # 维度相同则恒等映射
        else:
            self.register_buffer('res_fc', None)
        
        # 归一化层
        self.norm = None
        if norm is not None:
            self.norm = norm(out_dim)
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()  # 初始化线性层参数

    def forward(self, graph, feat):
        # 使用图g的局部作用域。
        with graph.local_scope():
            # 消息传递函数：复制源节点特征
            aggregate_fn = fn.copy_u('h', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            # 处理二分图，将特征分解为源节点和目标节点（expand_as_pair，用于处理二分图）。
            feat_src, feat_dst = expand_as_pair(feat, graph)

            # 计算归一化系数，并确保它和输入特征在同一个设备上
            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            # 将 norm 移动到 feat_src 所在的设备
            norm = norm.to(feat_src.device) 
            
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            # 调整形状以匹配源节点特征的维度
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm  # 源节点特征归一化

            # 消息聚合（邻居求和）：将源节点特征设置为'h'，然后进行图更新：聚合（copy_u）然后使用fn.sum求和（消息函数和聚合函数）
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            # 得到目标节点的聚合结果rst
            rst = graph.dstdata['h']
            
            # 线性变换
            rst = self.fc(rst)

            # if self._norm in ['right', 'both']:
            # 对目标节点的归一化也进行同样处理
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            # 将 norm 移动到 rst 所在的设备
            norm = norm.to(rst.device)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

            # 残差连接
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            # 应用归一化和激活函数
            if self.norm is not None:
                rst = self.norm(rst)

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

####################
#     GIN
####################
class GIN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 aggr="sum",
                 ):
        super(GIN, self).__init__()
        # 初始化参数
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()   # 存储GIN层
        self.activation = activation
        self.dropout = dropout

        last_activation = obtain_act(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        # 构建网络层
        if num_layers == 1:
            # 单层MLP
            apply_func = GIN_MLP(2, in_dim, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, norm=norm, activation=activation)
            self.layers.append(GINConv(in_dim, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))
        else:
            # input projection (no residual)
            self.layers.append(GINConv(
                in_dim, 
                num_hidden, 
                ApplyNodeFunc(GIN_MLP(2, in_dim, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                init_eps=0,
                learn_eps=learn_eps,
                residual=residual)
                )
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.layers.append(GINConv(
                    num_hidden, num_hidden, 
                    ApplyNodeFunc(GIN_MLP(2, num_hidden, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                    init_eps=0,
                    learn_eps=learn_eps,
                    residual=residual)
                )
            # output projection
            apply_func = GIN_MLP(2, num_hidden, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, activation=activation, norm=norm)

            self.layers.append(GINConv(num_hidden, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h)
            hidden_list.append(h)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GINConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 apply_func,
                 aggregator_type="sum",
                 init_eps=0,
                 learn_eps=False,
                 residual=False,
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim
        self.apply_func = apply_func    # MLP函数

        self._aggregator_type = aggregator_type # 聚合方式（sum/mean/max）
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
            
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))    # 可学习的epsilon
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

        # 残差连接配置
        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_u('h', 'm')

            # 聚合邻居特征
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))
            # 中心节点特征 + 聚合邻居
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                # 通过MLP
                rst = self.apply_func(rst)

            if self.res_fc is not None:
                # 残差连接
                rst = rst + self.res_fc(feat_dst)

            return rst


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp, norm="batchnorm", activation="relu"):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        norm_func = obtain_norm(norm)
        if norm_func is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm_func(self.mlp.output_dim)
        self.act = obtain_act(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class GIN_MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super(GIN_MLP, self).__init__()
        # 构建线性层、归一化层和激活函数
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(obtain_norm(norm)(hidden_dim))
                self.activations.append(obtain_act(activation))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)


####################
#     BWGNN
####################
class PolyConv(nn.Module):
    """多项式图卷积"""
    def __init__(self,
                 in_dim,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta     # 多项式系数
        self._k = len(self._theta)  # 多项式阶数
        self._in_dim = in_dim
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_dim, out_feats, bias)
        self.lin = lin


    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        # 未归一化的拉普拉斯算子: I - D^{-1/2}AD^{-1/2}
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            # 计算归一化系数
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            # 多项式卷积: Σ θ_k · (I - D^{-1/2}AD^{-1/2})^k
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        # 可选线性层
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h



def calculate_theta2(d):
    """计算Bernstein多项式系数"""
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        # 使用sympy计算多项式系数
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)    # 反转系数顺序
    return thetas


class BWGNN(nn.Module):
    """Bernstein-Wavelet GNN"""
    def __init__(self, in_dim, num_hidden, 
                 encoding=False, d=2, dropout=False):
        super(BWGNN, self).__init__()

        # encoder need activation, decode not
        self.is_encoder = encoding
        self.dropout = dropout

        self.thetas = calculate_theta2(d=d) # 获取多项式系数
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(num_hidden, num_hidden, self.thetas[i], lin=False)) # always no lin
        # 特征变换层
        self.linear = nn.Linear(in_dim, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)

        self.act = nn.ReLU()
        self.d = d

    def forward(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0]).to(g.device)
        for conv in self.conv:
            # 多尺度特征拼接
            h_final = torch.cat([h_final, conv(g, h)], -1)

        if self.is_encoder:
            h_final = self.act(h_final)
        # TODO: dropout?
        if self.dropout:
            raise NotImplementedError
        return h_final

####################
#     GraphSAGE
####################
class GraphSAGE(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 aggregator_type='mean',
                 encoding=False
                 ):
        super(GraphSAGE, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.sage_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

        # --- 核心修改：将 Norm 层独立出来 ---
        self.norms = nn.ModuleList()
        norm_layer_class = obtain_norm(norm)
        # --- 修改结束 ---

        last_activation = obtain_act(activation) if encoding else None
        
        if num_layers == 1:
            # --- 核心修改：移除 SAGEConv 中的 norm 参数 ---
            self.sage_layers.append(SAGEConv(
                in_dim, out_dim, aggregator_type, feat_drop=dropout,
                activation=last_activation
            ))
            if norm != 'none': self.norms.append(norm_layer_class(out_dim))
            # --- 修改结束 ---
        else:
            # Input layer
            self.sage_layers.append(SAGEConv(
                in_dim, num_hidden, aggregator_type, feat_drop=dropout,
                activation=obtain_act(activation)
            ))
            if norm != 'none': self.norms.append(norm_layer_class(num_hidden))

            # Hidden layers
            for l in range(1, num_layers - 1):
                self.sage_layers.append(SAGEConv(
                    num_hidden, num_hidden, aggregator_type, feat_drop=dropout,
                    activation=obtain_act(activation)
                ))
                if norm != 'none': self.norms.append(norm_layer_class(num_hidden))

            # Output layer
            self.sage_layers.append(SAGEConv(
                num_hidden, out_dim, aggregator_type, feat_drop=dropout,
                activation=last_activation
            ))
            if norm != 'none' and encoding: self.norms.append(norm_layer_class(out_dim))
        
        if residual:
            self.res_connections = nn.ModuleList()
            if num_layers == 1:
                self.res_connections.append(nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity())
            else:
                self.res_connections.append(nn.Linear(in_dim, num_hidden, bias=False) if in_dim != num_hidden else nn.Identity())
                for _ in range(num_layers - 2):
                    self.res_connections.append(nn.Identity())
                self.res_connections.append(nn.Linear(num_hidden, out_dim, bias=False) if num_hidden != out_dim else nn.Identity())

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        if inputs.dtype == torch.long:
            h = inputs.float()
        else:
            h = inputs
        
        hidden_list = []
        for l in range(self.num_layers):
            h_in = h
            h = F.dropout(h, p=self.dropout, training=self.training) # 将 dropout 移到卷积之前
            h = self.sage_layers[l](g, h)
            
            # --- 核心修改：手动应用残差和归一化 ---
            if self.residual:
                h = h + self.res_connections[l](h_in)
            
            # 只有在非输出层，或者输出层是编码器时，才应用 norm
            if l < self.num_layers - 1:
                if self.norms: h = self.norms[l](h)
            else: # last layer
                if self.norms and len(self.norms) == self.num_layers:
                    h = self.norms[l](h)
            # --- 修改结束 ---
            
            hidden_list.append(h)
        
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)