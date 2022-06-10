import torch
from torch import nn
import torch.nn.functional as F


def sagpool(x, adj, s, nodenum):
    _, perm = s.sort(dim=-1, descending=True)
    b = adj.size(0)
    perm = perm[:, :nodenum]
    print(perm)
    x = torch.cat([x[i, perm[i], :].unsqueeze(0) for i in range(b)])
    print(x.shape)
    adj = torch.cat([adj[i, perm[i], perm[i]].unsqueeze(0) for i in range(b)])
    print(adj.shape)


s = torch.tensor([list(range(0, 116)), list(range(4, 120))[::-1]]).float()
adj = torch.ones(2, 116, 116, dtype=torch.float)
x = torch.rand(2, 116, 3)

print(x.shape)
print(adj.shape)

sagpool(x, adj, s, 58)
