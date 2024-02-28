import torch
import torch.nn as nn
import torch
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling
import numpy as np
from torch.nn import init
from operator import itemgetter
import math
import torch.nn.functional as F
from torch.autograd import Variable


# 类间
class InterAgg(nn.Module):

    def __init__(self, feature_dim,
                 embed_dim, adj_lists, inter_agg
                 , cuda=True):
        super(InterAgg, self).__init__()

        self.dropout = 0.6
        self.adj_lists = adj_lists
        self.inter_agg = inter_agg

        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.cuda = cuda

        self.intra_agg1 = IntraAgg(self.feat_dim, cuda=self.cuda)
        self.intra_agg2 = IntraAgg(self.feat_dim, cuda=self.cuda)
        self.intra_agg3 = IntraAgg(self.feat_dim, cuda=self.cuda)
        # label predictor for similarity measure

        # initial filtering thresholds
        self.thresholds = [0.5, 0.5, 0.5]

        # the activation function used by attention mechanism
        self.leakyrelu = nn.LeakyReLU(0.2)

        # parameter used to transform node embeddings before inter-relation aggregation
        self.weight = nn.Parameter(torch.FloatTensor(190, self.embed_dim))
        init.xavier_uniform_(self.weight)

        # parameters used by attention layer
        self.a = nn.Parameter(torch.FloatTensor(2 * self.embed_dim, 1))
        init.xavier_uniform_(self.a)

        # initialize the parameter logs
        self.weights_log = []
        self.thresholds_log = [self.thresholds]
        self.relation_score_log = []


    def forward(self, nodes, features):
        self.features = features

        to_neighs = []
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
                                 set.union(*to_neighs[2], set(nodes)))

        if self.cuda:
            batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
        else:
            # batch_features = self.features(torch.LongTensor(list(unique_nodes)))
            batch_features = self.features[list(unique_nodes)]

        id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}


        r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
        r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
        r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

        r1_sample_num_list = [math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list]
        r2_sample_num_list = [math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list]
        r3_sample_num_list = [math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]

        r1_feats = self.intra_agg1.forward(nodes, r1_list, batch_features, id_mapping, r1_sample_num_list, features)
        r2_feats = self.intra_agg2.forward(nodes, r2_list, batch_features, id_mapping, r2_sample_num_list, features)
        r3_feats = self.intra_agg3.forward(nodes, r3_list, batch_features, id_mapping, r3_sample_num_list, features)

        neigh_feats = torch.cat((r1_feats, r2_feats, r3_feats), dim=0)

        if isinstance(nodes, list) and self.cuda == True:
            index = torch.LongTensor(nodes).cuda()
        else:
            index = torch.LongTensor(nodes)

        if self.cuda == True:
            self_feats = self.features(index)
        else:
            self_feats = self.features[list(index)]

        # number of nodes in a batch
        n = len(nodes)

        # inter-relation aggregation
        if isinstance(self_feats, tuple):
            self_feats = self_feats[0]
        combined, attention = att_inter_agg(len(self.adj_lists), self.leakyrelu, self_feats, neigh_feats,
                                            self.embed_dim,
                                            self.weight, self.a, n, self.dropout, self.training, self.cuda)

        return combined, [r1_feats, r2_feats, r3_feats]


# 类内
class IntraAgg(nn.Module):

    def __init__(self, feat_dim, cuda):
        super(IntraAgg, self).__init__()

        self.cuda = cuda
        self.feat_dim = feat_dim

    def forward(self, nodes, to_neighs_list, batch_features, id_mapping, sample_list, features):
        self.features = features
        # filer neighbors under given relation
        samp_neighs = filter_neighs_ada_threshold(nodes, batch_features, to_neighs_list, id_mapping,
                                                  sample_list)

        # find the unique nodes among batch nodes and the filtered neighbors
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # intra-relation aggregation only with sampled neighbors
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        # 平均聚合
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            # embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            embed_matrix = self.features[unique_nodes_list]

        if isinstance(embed_matrix, tuple):
            embed_matrix = embed_matrix[0]

        to_feats = mask.mm(embed_matrix)
        to_feats = F.relu(to_feats)
        return to_feats


# top-p采样
def filter_neighs_ada_threshold(nodes, batch_features, neighs_list, id_mapping, sample_list):
    samp_neighs = []

    # 计算中心节点特征和邻居节点特征的转置矩阵
    if (torch.is_tensor(nodes)):
        nodes = nodes.tolist()
    if isinstance(batch_features, tuple):
        batch_features = batch_features[0]

    center_features = batch_features[itemgetter(*nodes)(id_mapping), :]

    for i in range(len(center_features)):
        center = center_features[i]  # 中心节点特征
        neighbors = neighs_list[i]  # 邻居节点列表
        neighs_indices = neighs_list[i]

        neighbors_features = batch_features[itemgetter(*neighbors)(id_mapping), :]  # 邻居特征

        # 计算中心节点特征和邻居节点特征的转置矩阵
        if (len(neighs_indices) == 1):
            temp = neighbors_features.shape[0]
            neighbors_transposed = neighbors_features.view(temp)
            neighbors_transposed = neighbors_transposed.t()
        else:
            neighbors_transposed = torch.transpose(neighbors_features, 0, 1)

        # 计算中心节点与邻居节点之间的相似度
        similarity = torch.matmul(center, neighbors_transposed)
        center_norm = torch.norm(center)
        if len(neighs_indices) == 1:
            neighbors_norm = torch.norm(neighbors_features)
        else:
            neighbors_norm = torch.norm(neighbors_features, dim=1)
        similarity /= center_norm * neighbors_norm
        sorted_scores, sorted_indices = torch.sort(similarity, dim=0, descending=True)
        selected_indices = sorted_indices.tolist()
        num_sample = sample_list[i]
        if len(neighs_list[i]) > num_sample + 1:  # 邻居节点数量大于 采样 节点
            selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
        else:
            selected_neighs = neighs_indices
        samp_neighs.append(set(selected_neighs))
    return samp_neighs


def att_inter_agg(num_relations, att_layer, self_feats, neigh_feats, embed_dim, weight, a, n, dropout, training, cuda):
    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(neigh_feats, weight)

    combined = torch.cat((center_h.repeat(3, 1), neigh_h), dim=1)
    e = att_layer(combined.mm(a))
    attention = torch.cat((e[0:n, :], e[n:2 * n, :], e[2 * n:3 * n, :]), dim=1)
    ori_attention = F.softmax(attention, dim=1)  # 每行做Softmax，相当于每个节点做softmax
    attention = F.dropout(ori_attention, dropout, training=training)

    if cuda:
        aggregated = torch.zeros(size=(n, embed_dim)).cuda()
    else:
        aggregated = torch.zeros(size=(n, embed_dim))

    for r in range(num_relations):
        aggregated += torch.mul(attention[:, r].unsqueeze(1).repeat(1, embed_dim), neigh_h[r * n:(r + 1) * n, :])

    combined = F.relu((center_h + aggregated))

    att = F.softmax(torch.sum(ori_attention, dim=0), dim=0)

    return combined, att
