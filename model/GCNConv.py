import torch
from torch.nn import Parameter
from .init import glorot, zeros


## https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/dense_gcn_conv.html#DenseGCNConv
class DenseGCNConv(torch.nn.Module):
    r"""The graph convolutional operator from the `"Semi-supervised
        Classification with Graph Convolutional Networks"
        <https://arxiv.org/abs/1609.02907>`_ paper

        .. math::
            \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

        where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
        adjacency matrix with inserted self-loops and
        :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

        Args:
            in_channels (int): Size of each input sample.
            out_channels (int): Size of each output sample.
            improved (bool, optional): If set to :obj:`True`, the layer computes
                :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
                (default: :obj:`False`)
            cached (bool, optional): If set to :obj:`True`, the layer will cache
                the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
                \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
                cached version for further executions.
                This parameter should only be set to :obj:`True` in transductive
                learning scenarios. (default: :obj:`False`)
            bias (bool, optional): If set to :obj:`False`, the layer will not learn
                an additive bias. (default: :obj:`True`)
        """

    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
