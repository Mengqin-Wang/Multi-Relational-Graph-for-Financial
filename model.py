import numpy as np
from sklearn.preprocessing import StandardScaler

from layers import *

import torch
from torch.nn import functional as F

class FraudGCN(nn.Module):

    def __init__(self, feature, feature_len, num_classes, inter1, lstm_data, embed_dim, alpha, gamma, ):

        super(FraudGCN, self).__init__()
        self.inter1 = inter1
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(inter1.embed_dim, num_classes))

        self.lstm_data = lstm_data
        init.xavier_uniform_(self.weight)
        self.embed_dim = embed_dim

        self.alpha = alpha
        self.gamma = gamma

        self.linear1 = torch.nn.Linear(inter1.embed_dim + 32, 64)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, 32)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(32, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.feature = feature
        self.cuda = False

        self.weight = nn.Parameter(torch.FloatTensor(190, 32))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes,features):
        end, [r1_feats, r2_feats, r3_feats] = self.inter1(nodes,features)

        # data = torch.mm(features, self.weight)
        scaler = StandardScaler()
        data = scaler.fit_transform(self.feature)
        data = torch.tensor(data, dtype=torch.float32)
        if self.cuda:
            data = torch.mm(data.cuda(), self.weight)
        else:
            data = torch.mm(data, self.weight)
        # data_feature_MLP = data
        # temp = np.zeros(shape=(embeds1.size()[0], 32))
        # temp_features = data_feature_MLP.cpu().detach().numpy()
        # j = 0
        # for i in nodes:
        #     temp[j] = temp_features[i]
        #     j = j + 1
        # node_feature = torch.from_numpy(temp)
        node_feature = torch.from_numpy(data.cpu().detach().numpy()[nodes])

        if self.cuda == True:
            end = torch.cat((end, node_feature.cuda()), 1)
        else:
            end = torch.cat((end, node_feature), 1)
        end = end.to(torch.float32)

        x = self.linear1(end)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        scores = self.linear3(x)

        return scores, end, [r1_feats, r2_feats, r3_feats]

    def loss(self, nodes, labels,features):
        gnn_scores, end, [r1_feats, r2_feats, r3_feats] = self.forward(nodes,features)
        if type(labels) is np.ndarray:
            labels = torch.tensor(labels)
            labels = labels.type(torch.LongTensor)
        gnn_prob = nn.functional.softmax(gnn_scores, dim=1)
        if self.cuda:
            gnn_loss_xent = self.xent(gnn_scores.cuda(), labels.squeeze().cuda())
        else:
            gnn_loss_xent = self.xent(gnn_scores, labels.squeeze())
        if self.cuda:
            gnn_loss = sigmoid_focal_loss(gnn_loss_xent, gnn_scores, labels.t().cuda(), alpha=self.alpha,
                                          gamma=self.gamma)
        else:
            gnn_loss = sigmoid_focal_loss(gnn_loss_xent, gnn_scores, labels.t(), alpha=self.alpha, gamma=self.gamma)

        final_loss = gnn_loss

        return gnn_prob, final_loss, end, [r1_feats, r2_feats, r3_feats]




def sigmoid_focal_loss(
        gnn_loss_xent: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
        gamma: float,
        reduction: str = "sum",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.view([targets.size()[0], -1])
    targets = targets.float()
    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = gnn_loss_xent * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss
