import torch
import torch.nn as nn
import torch.nn.functional as F
from .GraphAttention import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0, alpha=0.2, nheads=5):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

        self.linear = nn.Linear(116 * nclass, nclass)

    def forward(self, x, adj):

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj)[0] for att in self.attentions], dim=2)
        attentions = [att(x, adj)[1] for att in self.attentions]
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = x.view(x.size()[0], -1)

        return self.linear(x), attentions
