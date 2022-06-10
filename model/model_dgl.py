import torch
from torch import nn
import dgl
import numpy as np
from dgl.nn import GATConv
import networkx as nk
import torch.nn.functional as F
import pickle
class GAT(nn.Module):#定义多层GAT模型
    def __init__(self, 
                num_layers,#层数
                in_dim,    #输入维度
                num_hidden,#隐藏层维度
                num_classes,#类别个数
                heads,#多头注意力的计算次数
                activation,#激活函数
                feat_drop,#特征层的丢弃率
                attn_drop,#注意力分数的丢弃率
                negative_slope,#LeakyReLU激活函数的负向参数
                residual):#是否使用残差网络结构
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0],
                                feat_drop, attn_drop, negative_slope, False, self.activation))
        #定义隐藏层
        for l in range(1, num_layers):
            #多头注意力 the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l], 
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        #输出层
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
    def forward(self, g,inputs): 
        h = inputs
        for l in range(self.num_layers):#隐藏层
                h = self.gat_layers[l](g, h).flatten(1)
        #输出层
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits
def getmodel(GAT): #定义函数实例化模型
        #定义模型参数
        num_heads = 8
        num_layers = 1
        num_out_heads =1
        heads = ([num_heads] * num_layers) + [num_out_heads]
        #实例化模型
        model = GAT(num_layers=num_layers, in_dim=3, num_hidden= 8,
                    num_classes = 2,
                    heads = ([num_heads] * num_layers) + [num_out_heads],#总的注意力头数
                    activation = F.elu, feat_drop=0.6, attn_drop=0.6,
                    negative_slope = 0.2, residual = True) #使用残差结构
        return model
if __name__ == "__main__":
    
        num_heads = 8
        num_layers = 1
        num_out_heads =1
        heads = ([num_heads] * num_layers) + [num_out_heads]
        #实例化模型
        model = GAT(num_layers=num_layers, in_dim=2048, num_hidden= 8,num_classes = 2,
                        heads = ([num_heads] * num_layers) + [num_out_heads],#总的注意力头数
                        activation = F.elu, feat_drop=0.6, attn_drop=0.6,
                        negative_slope = 0.2, residual = True) #使用残差结构
        print(model)
        #model = GAT(feat_dim=3, node_num=90, hidden_num=5, class_num=2)
        # print(model)
        # x = torch.tensor([list(range(0, 90))[::-1]]).float()
        # x = torch.cat([x, x, x], dim=-1)
        x=torch.randn(100,2048)
        print(x.shape)
        adj = nk.complete_graph(20)
        # adj.self_loop
        adj = dgl.from_networkx((adj))
        adj = dgl.add_self_loop(adj)
        print(adj)
        # adj = np.ones((2, 90, 90))
        with open("graph.pkl",'wb') as f:
                pickle.dump(adj,f)
        adj =dgl.batch([adj,adj,adj,adj,adj])
        print(adj)
        a = model(adj,x)
        print(a.shape)
