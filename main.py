import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm

from module import HeCo, Mp_encoder, Generator
from utils import load_data_mat, load_data_npz, set_params, eva, new_graph, normalize, get_A_r_flex, get_feature_dis, \
    Ncontrast

warnings.filterwarnings('ignore')





# random seed
# setup_seed(args.seed)

def cross_correlation(Z_v1, Z_v2):
    """
    calculate the cross-view correlation matrix S
    Args:
        Z_v1: the first view embedding
        Z_v2: the second view embedding
    Returns: S
    """
    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())


def off_diagonal(x):
    """
    off-diagonal elements of x
    Args:
        x: the input matrix
    Returns: the off-diagonal elements of x
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def correlation_reduction_loss(S):
    """
    the correlation reduction loss L: MSE for S and I (identical matrix)
    Args:
        S: the cross-view correlation matrix S
    Returns: L
    """
    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_dec():

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")

    # feats, adj_list, pos, knn_adj, label = load_data_mat(args.dataset, args.sc)
    feats, adj_list, pos, knn_adj, label = load_data_npz(args.dataset, args.sc)

    nb_classes = label.shape[-1]
    feats_dim = feats.shape[-1]
    P = int(len(adj_list))
    args.views = P

    knn_adj_hop = get_A_r_flex(knn_adj, args.order)

    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)
    print(args)

    model = HeCo(args.hidden_dim, feats_dim, args.feat_drop, args.attn_drop, P, nb_classes,
                 args.tau, args.lam, args)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        feats = feats.cuda()
        adj_list = [mp.cuda() for mp in adj_list]
        knn_adj_hop = knn_adj_hop.cuda()
        pos = pos.cuda()

    cnt_wait = 0
    best = 1e9
    loss = -1

    maxAcc_kmeans = -1
    best_nmi_kmeans = -1
    best_ari_kmeans = -1
    best_f1_kmeans = -1
    best_epoch_kmeans = 0

    z_mp, embeds_list, embed_q, q, contr_loss = model(feats, adj_list, pos)
    kmeans = KMeans(n_clusters=nb_classes, n_init=20)  # n_init：用不同的聚类中心初始化值运行算法的次数
    kmeans.fit_predict(z_mp.data.cpu().numpy())  # 训练并直接预测
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)  # kmeans.cluster_centers_：返回中心的坐标

    pbar = tqdm(range(args.nb_epochs))

    for epoch in pbar:
        model.train()

        if epoch % 3 == 0:  # dblp: 3  # acm: 3 freebase: 1 3

            z_mp, embeds_list, embed_q, q, contr_loss = model(feats, adj_list, pos)

            p = target_distribution(q.data)

            y_true = np.argmax(label, axis=1)
            kmeans = KMeans(n_clusters=nb_classes, n_init=20)
            y_pred_kmeans = kmeans.fit_predict(z_mp.data.cpu().numpy())
            acc_kmeans, nmi_kmeans, ari_kmeans, f1_kmeans = eva(y_true, y_pred_kmeans)

            if acc_kmeans > maxAcc_kmeans:
                maxAcc_kmeans = acc_kmeans
                best_nmi_kmeans = nmi_kmeans
                best_ari_kmeans = ari_kmeans
                best_f1_kmeans = f1_kmeans
                best_epoch_kmeans = epoch

            desc = "epoch:{} acc:{:.4f} nmi:{:.4f} ari:{:.4f} loss:{:.8f} maxAcc:{:.4f}" \
                   " best_nmi:{:.4f} best_ari:{:.4f} best_f1:{:.4f} best_epoch:{}".format(
                epoch, acc_kmeans, nmi_kmeans, ari_kmeans,
                loss, maxAcc_kmeans,
                best_nmi_kmeans,
                best_ari_kmeans,
                best_f1_kmeans,
                best_epoch_kmeans)
            pbar.set_description(desc)  # 相当于在当前长度的基础上 +1 的操作

        z_mp, embeds_list, embed_q, q, contr_loss = model(feats, adj_list, pos)

        # 传播正则化
        # az = torch.spmm(knn_adj.to('cuda'), z_mp)
        # p_output = F.softmax(az, dim=1)
        # q_output = F.softmax(z_mp, dim=1)
        # log_mean_output = ((p_output + q_output) / 2).log()
        # reg_loss = (F.kl_div(log_mean_output, p_output, reduction='batchmean') +
        #             F.kl_div(log_mean_output, q_output, reduction='batchmean')) / 2

        corr_loss = 0
        for i in range(len(embeds_list)):
            # corr_matrix = cross_correlation(z_mp, embeds_list[i])
            # corr_loss += correlation_reduction_loss(corr_matrix)
            #
            # corr_matrix = cross_correlation(embeds_list[i], z_mp)
            # corr_loss += correlation_reduction_loss(corr_matrix)
            #
            corr_matrix = cross_correlation(embeds_list[i].t(), z_mp.t())
            corr_loss += correlation_reduction_loss(corr_matrix)
            #
            # corr_matrix = cross_correlation(z_mp.t(), embeds_list[i].t())
            # corr_loss += correlation_reduction_loss(corr_matrix)

        z_mp_dis = get_feature_dis(z_mp)
        nContrast_loss = Ncontrast(z_mp_dis, knn_adj_hop, tau=args.parm_Ncontr)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        loss = args.parm_kl * kl_loss + args.parm_contr * contr_loss + args.parm_nContrast * nContrast_loss + args.parm_corr * corr_loss

        if loss < best:
            best = loss
            cnt_wait = 0

        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(best_epoch_kmeans, maxAcc_kmeans, best_nmi_kmeans, best_ari_kmeans, best_f1_kmeans, sep="---")

    return maxAcc_kmeans, best_nmi_kmeans, best_ari_kmeans, best_f1_kmeans


if __name__ == '__main__':

    # for parm_contr in [1]:
    #     for parm_nContrast in [0.1, 1, 10]:  # 0.001, 0.01,
    #         for parm_corr in [0.001, 0.01, 0.1, 1, 10]:  # 0.001, 0.01,
    #             for parm_Ncontr in [0.0001, 0.001, 0.01, 0.1, 1]:
    #                 for i in range(3):
    #
    #                     print(f"第{i + 1}次循环！！")
    #                     args.parm_contr = parm_contr
    #                     args.parm_nContrast = parm_nContrast
    #                     args.parm_corr = parm_corr
    #                     args.parm_Ncontr = parm_Ncontr
    #                     train_dec()
    #
    #                 with open(f'parameter_{args.dataset}.txt', 'a+') as f:
    #
    #                     f.write('\n')
    #                     f.flush()
    #
    #                 f.close()

    # for nContrast in [0.01, 0.1, 1, 10]:
    #     for corr in [0.001, 0.01, 0.1, 1, 10]:
    #         for i in range(3):
    #             args.parm_nContrast = nContrast
    #             args.parm_corr = corr
    #             print(f"第{i + 1}次循环")
    #             train_dec()
    #
    #         with open(f'temp_{args.dataset}.txt', 'a+') as f:
    #
    #             f.write('\n')
    #             f.flush()
    #
    #         f.close()

    # epochs = 10
    # for dataset in ['acm', 'dblp', 'freebase', 'aminer']:
    #     args = set_params(dataset)
    #     for parm_contr in [0.0001, 0.0001, 0.001, 0.001, 0.1]:  #  [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
    #         epoch_acc = 0
    #         epoch_nmi = 0
    #         epoch_ari = 0
    #         epoch_f1 = 0
    #         for i in range(epochs):
    #             print(f"第{i + 1}次循环 → {parm_contr}")
    #             args.parm_contr = parm_contr
    #             # args.dataset = dataset
    #             acc, nmi, ari, f1 = train_dec()
    #             epoch_acc += acc
    #             epoch_nmi += nmi
    #             epoch_ari += ari
    #             epoch_f1 += f1
    #
    #         with open(f'parm_contr.txt', 'a+') as f:
    #             f.write(
    #                 f"dataset: {args.dataset} -- parm_contr: {args.parm_contr} "
    #                 f"-- mean_acc: {epoch_acc / epochs} -- mean_nmi: {epoch_nmi / epochs} "
    #                 f"-- mean_ari: {epoch_ari / epochs} -- mean_f1: {epoch_f1 / epochs} ")
    #             f.write('\n')
    #             f.flush()
    #
    #         f.close()

    args = set_params("dblp")
    train_dec()

    # train_gan()
