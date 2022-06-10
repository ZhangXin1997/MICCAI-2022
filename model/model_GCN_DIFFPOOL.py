import torch
from torch import nn
import torch.nn.functional as F

from .GCNConv import DenseGCNConv
from .DiffPool import dense_diff_pool


class GCN(nn.Module):
    def __init__(self, feat_dim, node_num, assign_ratio):
        super(GCN, self).__init__()

        assign_dim = node_num
        self.bn = True
        ## conv_block1
        self.conv1 = DenseGCNConv(feat_dim, 4)
        self.conv2 = DenseGCNConv(4, 2)

        # ## diff_pool_block1
        # assign_dim = int(assign_dim * assign_ratio)
        # self.conv1_pool1 = DenseGCNConv(feat_dim, 4)
        # self.conv2_pool1 = DenseGCNConv(4, assign_dim)
        # self.linear_pool1 = nn.Linear(assign_dim + 4, assign_dim)

        ## conv_block2
        self.conv3 = DenseGCNConv(6, 4)
        self.conv4 = DenseGCNConv(4, 2)

        # ## diff_pool_block2
        # assign_dim = int(assign_dim * assign_ratio)
        # self.conv1_pool2 = DenseGCNConv(6, 4)
        # self.conv2_pool2 = DenseGCNConv(4, assign_dim)
        # self.linear_pool2 = nn.Linear(assign_dim + 4, assign_dim)

        ## conv_block3
        self.conv5 = DenseGCNConv(6, 4)
        self.conv6 = DenseGCNConv(4, 2)

        # ## diff_pool_block3
        # assign_dim = int(assign_dim * assign_ratio)
        # self.conv1_pool3 = DenseGCNConv(6, 4)
        # self.conv2_pool3 = DenseGCNConv(4, assign_dim)
        # self.linear_pool3 = nn.Linear(assign_dim + 4, assign_dim)

        self.conv7 = DenseGCNConv(6, 4)
        self.conv8 = DenseGCNConv(4, 2)

        self.linear = nn.Linear(4 * 6, 3)

    def apply_bn(self, x):
        '''
        Batch normalization of 3D tensor x
        '''
        if self.bn:
            bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
            return bn_module(x)
        else:
            return x

    def forward(self, x, adj):
        loss = 0
        out_all = []
        ## conv_block1
        x_conv1 = self.apply_bn(F.relu(self.conv1(x, adj)))
        x_conv2 = self.apply_bn(F.relu(self.conv2(x_conv1, adj)))
        x_conv2 = torch.cat([x_conv1, x_conv2], dim=-1)
        out_all.append(x_conv2.max(dim=1)[0])

        # ## diffpool_block1
        # x_pool1 = self.apply_bn(F.relu(self.conv1_pool1(x, adj)))
        # x_pool2 = self.apply_bn(self.conv2_pool1(x_pool1, adj))
        # x_pool2 = torch.cat([x_pool1, x_pool2], dim=-1)
        # s = self.linear_pool1(x_pool2)
        # x, adj, link_loss, ent_loss = dense_diff_pool(x_conv2, adj, s)
        # loss = loss + link_loss + ent_loss

        ## conv_block2
        x_conv1 = self.apply_bn(F.relu(self.conv3(x_conv2, adj)))
        x_conv2 = self.apply_bn(F.relu(self.conv2(x_conv1, adj)))
        x_conv2 = torch.cat([x_conv1, x_conv2], dim=-1)
        out_all.append(x_conv2.max(dim=1)[0])

        # ## diffpool_block2
        # x_pool1 = self.apply_bn(F.relu(self.conv1_pool2(x, adj)))
        # x_pool2 = self.apply_bn(self.conv2_pool2(x_pool1, adj))
        # x_pool2 = torch.cat([x_pool1, x_pool2], dim=-1)
        # s = self.linear_pool2(x_pool2)
        # x, adj, link_loss, ent_loss = dense_diff_pool(x_conv2, adj, s)
        # loss = loss + link_loss + ent_loss

        ## conv_block3
        x_conv1 = self.apply_bn(F.relu(self.conv5(x_conv2, adj)))
        x_conv2 = self.apply_bn(F.relu(self.conv2(x_conv1, adj)))
        x_conv2 = torch.cat([x_conv1, x_conv2], dim=-1)
        out_all.append(x_conv2.max(dim=1)[0])

        # ## diffpool_block2
        # x_pool1 = self.apply_bn(F.relu(self.conv1_pool3(x, adj)))
        # x_pool2 = self.apply_bn(self.conv2_pool3(x_pool1, adj))
        # x_pool2 = torch.cat([x_pool1, x_pool2], dim=-1)
        # s = self.linear_pool3(x_pool2)
        # x, adj, link_loss, ent_loss = dense_diff_pool(x_conv2, adj, s)
        # loss = loss + link_loss + ent_loss

        x_conv1 = self.apply_bn(F.relu(self.conv7(x_conv2, adj)))
        x_conv2 = self.apply_bn(F.relu(self.conv2(x_conv1, adj)))
        x_conv2 = torch.cat([x_conv1, x_conv2], dim=-1)
        out_all.append(x_conv2.max(dim=1)[0])
        # out_all.append(x_conv2.view(x_conv2.size()[0], -1))
        # out = x_conv2.view(x_conv2.size()[0], -1)
        out = torch.cat(out_all, dim=-1)
        ypred = self.linear(out)

        return ypred, loss


if __name__ == "__main__":
    model = GCN(feat_dim=3, node_num=90, assign_ratio=0.5)
    print(model)
    x = torch.tensor([list(range(0, 90)), list(range(4, 94))[::-1]]).float().unsqueeze(-1)
    x = torch.cat([x, x, x], dim=-1)
    adj = torch.ones(2, 90, 90, dtype=torch.float)

    a = model(x, adj)
