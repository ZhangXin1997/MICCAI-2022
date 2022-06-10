import torch
from torch import nn
import torch.nn.functional as F
from model.GraphAttention import GraphAttentionLayer
from model.DiffPool import dense_diff_pool
# from GraphAttention import GraphAttentionLayer
# from DiffPool import dense_diff_pool

class GAN(nn.Module):
    def __init__(self, feat_dim, node_num, hidden_num, class_num, nheads,dropout=0.5): #2
        super(GAN, self).__init__()
        self.class_num = class_num
        assign_dim = node_num
        self.bn = False
        self.dropout = dropout

        # self.linear_feature = nn.Linear(feat_dim, 32)

        self.attentions = [GraphAttentionLayer(feat_dim, hidden_num, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hidden_num * nheads, hidden_num * nheads, concat=False)

        # ## diff_pool_block1
        # self.conv_pool1 = GraphAttentionLayer(hidden_num * nheads, 32) #16
        # # self.linear_pool1 = nn.Linear(32, 32)

        # # ## diff_pool_block2
        # self.conv_pool2 = GraphAttentionLayer(hidden_num * nheads, 4) #2
        # self.linear_pool2 = nn.Linear(4, 4)
        self.pooling = nn.MaxPool2d((20,1))
        # self.pooling = nn.AvgPool2d((20,1))
        self.linear = nn.Linear(hidden_num * nheads, self.class_num)

    def apply_bn(self, x):
        '''
        Batch normalization of 3D tensor x
        '''
        if self.bn:
            bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
            return bn_module(x)
        else:
            return x

    def forward(self, x,x_neg, adj,adj_neg):
        loss = 0

        # x = self.linear_feature(x)

        x = F.dropout(x, self.dropout, training=self.training)
        attentions = [att(x, adj)[1] for att in self.attentions]
        # print(len(attentions))
        # print(attentions[0])
        x = torch.cat([att(x, adj)[0] for att in self.attentions], dim=2)
        # print(x.shape)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj)[0])  # b, 116, 15
        # print(x.shape)
        # ## diffpool_block1
        # s = self.apply_bn(self.conv_pool1(x, adj)[0])
        # #x_pool1 = self.apply_bn(self.conv_pool1(x, adj)[0])
        # # s = self.linear_pool1(x_pool1)
        # x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s)  # b, 32, 15
        # loss = loss + link_loss

        # ## diffpool_block2
        # s = self.apply_bn(self.conv_pool2(x, adj)[0])
        # # x_pool2 = self.apply_bn(self.conv_pool2(x, adj)[0])
        # # s = self.linear_pool2(x_pool2)
        # x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s)  # b, 4, 15
        # loss = loss + link_loss

        # out = x.view(x.size()[0], -1)
        out = self.pooling(x).view(x.size()[0],-1)
        out = F.tanh(out)

        ################
        x_neg = F.dropout(x_neg, self.dropout, training=self.training)
        attentions = [att(x_neg, adj_neg)[1] for att in self.attentions]
        x_neg = torch.cat([att(x_neg, adj_neg)[0] for att in self.attentions], dim=2)

        # x_neg = F.dropout(x_neg, self.dropout, training=self.training)
        x_neg = F.elu(self.out_att(x_neg, adj_neg)[0])  # b, 116, 15

        # ## diffpool_block1
        # s = self.apply_bn(self.conv_pool1(x_neg, adj_neg)[0])
        # #x_neg_pool1 = self.apply_bn(self.conv_pool1(x_neg, adj_neg)[0])
        # # s = self.linear_pool1(x_neg_pool1)
        # x_neg, adj_neg, link_loss, ent_loss = dense_diff_pool(x_neg, adj_neg, s)  # b, 32, 15
        # loss = loss + link_loss
        # # print(link_loss)
        # ## diffpool_block2
        # s = self.apply_bn(self.conv_pool2(x_neg, adj_neg)[0])
        # # x_neg_pool2 = self.apply_bn(self.conv_pool2(x_neg, adj_neg)[0])
        # # s = self.linear_pool2(x_neg_pool2)
        # x_neg, adj_neg, link_loss, ent_loss = dense_diff_pool(x_neg, adj_neg, s)  # b, 4, 15
        # loss = loss + link_loss

        # out_neg = x_neg.view(x_neg.size()[0], -1)
        out_neg = self.pooling(x_neg).view(x_neg.size()[0],-1)
        out_neg = F.tanh(out_neg)
        ################
        # out = torch.mean(x, dim=1)


        cos_similarity = torch.cosine_similarity(out,out_neg,dim=1)

        ypred = self.linear(out)

        return ypred, cos_similarity, out,out_neg


if __name__ == "__main__":
    #model = GAN(feat_dim=3, node_num=90, assign_ratio=0.5)
    model = GAN(feat_dim=1024, node_num=20, hidden_num=5, class_num=2,nheads=8)
    # print(model)
    x = torch.randn(5,20,1024)
    x_neg = torch.randn(5,20,1024)
    adj = torch.ones(5, 20, 20, dtype=torch.float)
    adj_neg = torch.ones(5, 20, 20, dtype=torch.float)
    a,b,c,d = model(x, x_neg,adj,adj_neg)
    print(a)
    print(b)
    print(c)
    # print(d.shape)
