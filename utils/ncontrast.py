import torch


def get_A_r_flex(adj, r, cumulative=False):
    adj_d = adj.to_dense()
    adj_c = adj_d           # A1, A2, A3 .....
    adj_label = adj_d

    for i in range(r-1):
        adj_c = adj_c@adj_d
        adj_label = adj_label + adj_c if cumulative else adj_c
    return adj_label


def get_feature_dis(x):
    # x: batch_size x nhid
    # x_dis(i,j): item means the similarity between x(i) and x(j).
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis


def Ncontrast(x_dis, adj_label, tau = 1):
    # compute the Ncontrast loss
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss