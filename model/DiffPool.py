import torch
from torch import nn
import torch.nn.functional as F

# from GCNConv import DenseGCNConv
# from GraphConv import DenseGraphConv
EPS = 1e-15

def dense_diff_pool(x, adj, s, mask=None):
    r"""Differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns pooled node feature matrix, coarsened adjacency matrix and two
    auxiliary objectives: (1) The link prediction loss

    .. math::
        \mathcal{L}_{LP} = {\| \mathbf{A} -
        \mathrm{softmax}(\mathbf{S}) {\mathrm{softmax}(\mathbf{S})}^{\top}
        \|}_F,

    and the entropy regularization

    .. math::
        \mathcal{L}_E = \frac{1}{N} \sum_{n=1}^N H(\mathbf{S}_n).

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
            \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

    return out, out_adj, link_loss, ent_loss

# class DIFFPool(nn.Module):
#     def __init__(self, assign_input_dim, assign_hidden_dim, assign_dim):
#         super(DIFFPool, self).__init__()
#         self.assign_conv1 = DenseGraphConv(assign_input_dim, assign_hidden_dim)
#         self.assign_conv2 = DenseGraphConv(assign_hidden_dim, assign_dim)
#         self.linear = nn.Linear(assign_hidden_dim + assign_dim, assign_dim)
#
#     def forward(self, x, adj):
#         assign_tensor1 = F.relu(self.assign_conv1(x, adj))
#         assign_tensor1 = self.apply_bn(assign_tensor1)
#         assign_tensor2 = self.assign_conv2(assign_tensor1, adj)
#         assign_tensor2 = self.apply_bn(assign_tensor2)
#
#         x_tensor = torch.cat([assign_tensor1, assign_tensor2], dim=-1)
#         assign_tensor = self.linear(x_tensor)
#
#         return self.diff_pool(x, adj, assign_tensor)
#
#     def apply_bn(self, x):
#         '''
#         Batch normalization of 3D tensor x
#         '''
#         bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
#         return bn_module(x)
#
#     def diff_pool(self, x, adj, s, mask=None):
#         ## https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/dense/diff_pool.py
#         r"""Differentiable pooling operator from the `"Hierarchical Graph
#         Representation Learning with Differentiable Pooling"
#         <https://arxiv.org/abs/1806.08804>`_ paper
#         .. math::
#             \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
#             \mathbf{X}
#             \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
#             \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})
#         based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
#         \times N \times C}`.
#         Returns pooled node feature matrix, coarsened adjacency matrix and two
#         auxiliary objectives: (1) The link prediction loss
#         .. math::
#             \mathcal{L}_{LP} = {\| \mathbf{A} -
#             \mathrm{softmax}(\mathbf{S}) {\mathrm{softmax}(\mathbf{S})}^{\top}
#             \|}_F,
#         and the entropy regularization
#         .. math::
#             \mathcal{L}_E = \frac{1}{N} \sum_{n=1}^N H(\mathbf{S}_n).
#         Args:
#             x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
#                 \times N \times F}` with batch-size :math:`B`, (maximum)
#                 number of nodes :math:`N` for each graph, and feature dimension
#                 :math:`F`.
#             adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
#                 \times N \times N}`.
#             s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
#                 \times N \times C}` with number of clusters :math:`C`. The softmax
#                 does not have to be applied beforehand, since it is executed
#                 within this method.
#             mask (BoolTensor, optional): Mask matrix
#                 :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
#                 the valid nodes for each graph. (default: :obj:`None`)
#         :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
#             :class:`Tensor`)
#         """
#
#         x = x.unsqueeze(0) if x.dim() == 2 else x
#         adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
#         s = s.unsqueeze(0) if s.dim() == 2 else s
#
#         batch_size, num_nodes, _ = x.size()
#
#         s = torch.softmax(s, dim=-1)
#
#         if mask is not None:
#             mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
#             x, s = x * mask, s * mask
#
#         out = torch.matmul(s.transpose(1, 2), x)
#         out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
#
#         link_loss = adj - torch.matmul(s, s.transpose(1, 2))
#         link_loss = torch.norm(link_loss, p=2)
#         link_loss = link_loss / adj.numel()
#
#         ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()
#
#         return out, out_adj, link_loss, ent_loss
