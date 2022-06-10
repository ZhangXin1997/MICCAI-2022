import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)  # [b, N, out_features]
        b = h.size()[0]
        N = h.size()[1]
        a_input = torch.cat([h.repeat(1, 1, N).view(b, N * N, -1), h.repeat(1, N, 1)], dim=2).view(b, N, -1,
                                                                                                   2 * self.out_features)

        # [N, N, 2*out_features]
        a = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # [N, N, 1] => [N, N]
        e = adj * a
        # print(attention.shape)
        # zero_vec = -9e15 * torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        e = F.softmax(e, dim=2)
        attention = F.dropout(e, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime), e
        else:
            return h_prime, e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == "__main__":
    model = GraphAttentionLayer(3, 3, concat=False)
    print(model)
    x = torch.tensor([list(range(0, 116)), list(range(4, 120))[::-1]]).float().unsqueeze(-1)
    x = torch.cat([x, x, x], dim=-1)
    adj = torch.ones(2, 116, 116, dtype=torch.float)
    print(adj.shape, x.shape)
    a = model(x, adj)
    print(a.shape)
