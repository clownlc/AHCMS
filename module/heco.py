import torch
import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder
from utils import LogReg
from torch.nn.parameter import Parameter
from .contrast import Contrast


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


class HeCo(nn.Module):
    def __init__(self, hidden_dim, feats_dim, feat_drop, attn_drop, P, nb_classes,
                tau, lam, args, v=1):
        super(HeCo, self).__init__()
        self.hidden_dim = hidden_dim
        self.v = v

        self.fc = nn.Linear(feats_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = Mp_encoder(P, feats_dim, hidden_dim, attn_drop, args)  # P：元路径个数
        self.logist = LogReg(hidden_dim, nb_classes)
        self.contrast = Contrast(hidden_dim, tau, lam)

        self.cluster_layer = Parameter(torch.Tensor(nb_classes, hidden_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, feats, mps, pos):
        # h = F.elu(self.feat_drop(self.fc(feats)))  # 将不同类型的节点特征维度投影到同一维度
        z_mp, embeds_list = self.mp(feats, mps)  # 考虑元路径视图，返回融合各个元路径的低维表征

        q = 1.0 / (1.0 + torch.sum(torch.pow(z_mp.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        embeds_q = 0

        loss = self.contrast(z_mp, embeds_list, pos)

        return z_mp, embeds_list, embeds_q, q, loss

    def get_embeds(self, feats, mps):
        z_mp, embeds_list = self.mp(feats, mps)
        return z_mp.detach()

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):

        batch_size = x.shape[0]
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)

        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        if sym:

            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            #    print(pos_sim,sim_matrix.sum(dim=0))
            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
        else:
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()

        return loss

