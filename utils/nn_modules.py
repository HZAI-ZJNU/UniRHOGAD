import torch
import dgl
import torch.nn as nn
from itertools import product
from functools import reduce

# ======================================================================
#   MLP: 通用多层感知机
# ======================================================================
class MLP(nn.Module):
    """
    一个通用的多层感知机模块。
    """
    def __init__(self, in_feats, h_feats, out_feats, num_layers=2, dropout_rate=0.5, activation='ReLU', output_activation=False):
        """
        Args:
            in_feats (int): 输入特征维度。
            h_feats (int): 隐藏层维度。
            out_feats (int): 输出特征维度。
            num_layers (int): 总层数 (包括输入和输出层)。
            dropout_rate (float): Dropout 比例。
            activation (str): 隐藏层激活函数。
            output_activation (bool): 是否在输出层后应用激活函数。
        """
        super().__init__()
        self.layers = nn.ModuleList()
        try:
            act_fn = getattr(nn, activation)()
        except AttributeError:
            print(f"Activation function '{activation}' not found in torch.nn, defaulting to ReLU.")
            act_fn = nn.ReLU()
        
        if num_layers == 1:
            # 如果只有一层，直接从输入映射到输出
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            # 输入层
            self.layers.append(nn.Linear(in_feats, h_feats))
            self.layers.append(act_fn)
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            
            # 隐藏层
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(h_feats, h_feats))
                self.layers.append(act_fn)
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))
            
            # 输出层
            self.layers.append(nn.Linear(h_feats, out_feats))

        if output_activation:
            self.layers.append(act_fn)
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        纯粹的张量到张量的前向传播。
        
        Args:
            h (torch.Tensor): 输入的特征张量。
        
        Returns:
            torch.Tensor: 输出的特征张量。
        """
        for layer in self.layers:
            h = layer(h)
        return h
    

# ======================================================================
#   GraphStitchHead: 融合-预测头
# ======================================================================
class GraphStitchHead(nn.Module):
    """
    一个独立的模块，负责一种特定的 cross_mode 的信息融合和最终预测。
    它接收所有分支的图级别表示，并根据自己的路由策略进行融合和预测。
    """
    def __init__(self, cross_mode: str, embed_dim: int, num_classes: int, mlp_layers: int = 2, dropout_rate: float = 0.5, activation: str = 'ReLU'):
        """
        Args:
            cross_mode (str): 定义融合策略, e.g., "ng2ng".
            embed_dim (int): 输入的嵌入维度。
            num_classes (int): 最终分类器的输出类别数。
            **mlp_kwargs: 传递给内部MLP分类器的参数 (num_layers, dropout_rate, etc.)
        """
        super().__init__()
        self.cross_mode = cross_mode
        try:
            input_route_str, output_route_str = cross_mode.split('2')
            self.input_route = list(input_route_str)
            self.output_route = list(output_route_str)
        except ValueError:
            raise ValueError(f"Invalid cross_mode format: '{cross_mode}'. Expected format is 'source2target', e.g., 'ng2ng'.")
        
        # 1. GraphStitch 融合权重
        self.stitch_weights = nn.ParameterDict({
            f"{o_task}_from_{i_task}": nn.Parameter(torch.randn(1))
            for o_task in self.output_route for i_task in self.input_route
        })
        
        # 2. 最终的 MLP 预测器
        self.predictors = nn.ModuleDict({
            task: MLP(embed_dim, embed_dim, num_classes, 
                      num_layers=mlp_layers, 
                      dropout_rate=dropout_rate, 
                      activation=activation)
            for task in self.output_route
        })

    def forward(self, branch_representations: dict, is_single_graph: bool = False,
                 graph_for_node_task: dgl.DGLGraph=None) -> dict:

        logits_dict = {}


        if is_single_graph:
            # --- 单图场景：独立预测，不进行跨任务融合 ---
            for task, rep in branch_representations.items():
                if task in self.predictors:

                    logits_dict[task] = self.predictors[task](rep)

        else:
            # --- 多图场景：执行带有广播的GraphStitch融合逻辑 ---
            for o_task in self.output_route:
                if o_task not in self.predictors: continue

                # 1. 收集所有有效的输入源表示
                # inputs_to_fuse 是一个列表，存放加权后的表示张量
                inputs_to_fuse = []
                for i_task in self.input_route:
                    if i_task in branch_representations:
                        # 获取输入源的表示
                        source_rep = branch_representations[i_task]
                        
                        # 获取融合权重
                        weight = self.stitch_weights[f"{o_task}_from_{i_task}"]
                        
                        # 检查是否需要维度对齐
                        # 目标表示的形状，如果输出任务也在输入中，就用它，否则用源的形状（说明是n->n或g->g）
                        target_shape_ref = branch_representations.get(o_task, source_rep)
                        if source_rep.shape[0] != target_shape_ref.shape[0]:
                            # 维度不匹配，通常是 n <-> g 的情况
                            if o_task == 'n' and i_task == 'g': # g -> n, 需要广播
                                if graph_for_node_task is None: raise ValueError("Graph context needed for g->n fusion.")
                                aligned_rep = dgl.broadcast_nodes(graph_for_node_task, source_rep)
                            elif o_task == 'g' and i_task == 'n': # n -> g, 需要池化
                                if graph_for_node_task is None: raise ValueError("Graph context needed for n->g fusion.")
                                temp_feat_name = f"_temp_feat_{i_task}_to_{o_task}"
                                graph_for_node_task.ndata[temp_feat_name] = source_rep
                                aligned_rep = dgl.mean_nodes(graph_for_node_task, feat=temp_feat_name)
                                del graph_for_node_task.ndata[temp_feat_name] # 清理
                            else:
                                # 其他不匹配情况暂不处理
                                continue
                        else:
                            # 维度匹配，直接使用
                            aligned_rep = source_rep
                            
                        inputs_to_fuse.append(weight * aligned_rep)

                # 2. 如果没有任何有效的输入源，则跳过此任务
                if not inputs_to_fuse:
                    continue

                # 3. 融合所有加权后的输入表示
                fused_rep = reduce(torch.add, inputs_to_fuse)
                
                # 4. 可选的残差连接：如果输出任务本身也是一个输入源，可以添加残差
                if o_task in self.input_route and o_task in branch_representations:
                    fused_rep = fused_rep + branch_representations[o_task]

                # 5. 送入预测器
                logits_dict[o_task] = self.predictors[o_task](fused_rep)
        

        return logits_dict

    
# ======================================================================
#   GraphStitchHead_SimpleFusion: 带有简单融合的消融实验版本
# ======================================================================
class GraphStitchHead_SimpleFusion(GraphStitchHead):
    """
    GraphStitchHead的消融实验版本，实现了多种简单的信息融合策略。
    它继承自原始的 GraphStitchHead，只重写 forward 方法以处理单图场景。
    """
    def __init__(self, fusion_mode='concat', *args, **kwargs):
        # 消费掉 fusion_mode, 然后安全地调用父类
        super().__init__(*args, **kwargs)
        
        self.fusion_mode = fusion_mode
        embed_dim = kwargs.get('embed_dim', 0)
        num_classes = kwargs.get('num_classes', 2)
        
        if self.fusion_mode == 'concat':
            # 拼接模式下，输入维度加倍
            fused_embed_dim = embed_dim * 2
            self.fused_predictors = nn.ModuleDict({
                task: MLP(fused_embed_dim, embed_dim, num_classes, 
                          num_layers=kwargs.get('mlp_layers', 2), 
                          dropout_rate=kwargs.get('dropout_rate', 0.5), 
                          activation=kwargs.get('activation', 'ReLU'))
                for task in self.output_route
            })
        elif self.fusion_mode == 'attention':
            # 注意力模式下，我们学习一个权重 alpha
            self.attention_net = nn.ModuleDict()
            for task in self.output_route:
                # 输入是拼接的 [local_rep, global_context]，输出是单个标量 alpha
                self.attention_net[task] = nn.Sequential(
                    nn.Linear(embed_dim * 2, embed_dim),
                    nn.Tanh(),
                    nn.Linear(embed_dim, 1),
                    nn.Sigmoid()
                )
                
    def forward(self, branch_representations: dict, is_single_graph: bool = False,
                 graph_for_node_task: dgl.DGLGraph=None) -> dict:
        """
        重写的 forward 方法。如果是在单图场景，则执行融合逻辑。
        否则，调用父类的原始方法处理多图场景。
        """
        if not is_single_graph:
            # 对于多图场景，行为与父类完全相同
            return super().forward(branch_representations, is_single_graph, graph_for_node_task)

        # --- 单图场景：执行融合逻辑 ---
        logits_dict = {}
        
        node_rep = branch_representations.get('n')
        edge_rep = branch_representations.get('e')
        
        # 池化表示以获得图级别的上下文
        graph_context_from_edges = edge_rep.mean(dim=0, keepdim=True) if edge_rep is not None else None
        graph_context_from_nodes = node_rep.mean(dim=0, keepdim=True) if node_rep is not None else None

        # 1. 预测节点任务
        if 'n' in self.predictors and node_rep is not None:
            if graph_context_from_edges is not None:
                context_expanded = graph_context_from_edges.expand(node_rep.shape[0], -1)
                
                if self.fusion_mode == 'concat':
                    fused_rep = torch.cat([node_rep, context_expanded], dim=1)
                    logits_dict['n'] = self.fused_predictors['n'](fused_rep)
                elif self.fusion_mode == 'attention':
                    attention_input = torch.cat([node_rep, context_expanded], dim=1)
                    alpha = self.attention_net['n'](attention_input)
                    fused_rep = (1 - alpha) * node_rep + alpha * context_expanded
                    # 注意力模式下维度不变，复用父类的 predictors
                    logits_dict['n'] = self.predictors['n'](fused_rep)
            else:
                # 如果没有全局上下文，则独立预测
                logits_dict['n'] = self.predictors['n'](node_rep)

        # 2. 预测边任务 (逻辑与节点任务类似)
        if 'e' in self.predictors and edge_rep is not None:
            if graph_context_from_nodes is not None:
                context_expanded = graph_context_from_nodes.expand(edge_rep.shape[0], -1)

                if self.fusion_mode == 'concat':
                    fused_rep = torch.cat([edge_rep, context_expanded], dim=1)
                    logits_dict['e'] = self.fused_predictors['e'](fused_rep)
                elif self.fusion_mode == 'attention':
                    attention_input = torch.cat([edge_rep, context_expanded], dim=1)
                    alpha = self.attention_net['e'](attention_input)
                    fused_rep = (1 - alpha) * edge_rep + alpha * context_expanded
                    logits_dict['e'] = self.predictors['e'](fused_rep)
            else:
                # 如果没有全局上下文，则独立预测
                logits_dict['e'] = self.predictors['e'](edge_rep)

        return logits_dict