import pickle as pkl
import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


####################################################
# This tool is to generate positive set with a thre-
# shold "pos_num".
# dataset  pos_num
# acm      5 1895 1 1085.788753421249
# dblp     1000 4057 579 2206.4572344096623
# aminer   15 216 1 21.583790371724557
# freebase 80 451 1 74.75200458190149
# amazon   100 1210 1 163.2737173599265
#
#
# Notice: The best pos_num of acm is 7 reported in 
# paper, but we find there is no much difference 
# between 5 and 7 in practice.
####################################################
def get_pos_acm_npz(sc=0):
    path = "../data/acm/"

    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")

    pap = normalize_adj(pap)
    psp = normalize_adj(psp)

    pos_num = 5
    p = 4019
    pap = pap / pap.sum(axis=-1).reshape(-1, 1)
    psp = psp / psp.sum(axis=-1).reshape(-1, 1)
    all = (pap + psp).A.astype("float32")
    all_ = (all > 0).sum(-1)
    print(pos_num, all_.max(), all_.min(), all_.mean())  # 5 1895 1 1085.788753421249

    pos = np.zeros((p, p))
    k = 0
    for i in range(len(all)):
        one = all[i].nonzero()[0]  # 输出每一行不为零的数的索引（即节点i的邻居节点）
        if len(one) > pos_num:
            oo = np.argsort(-all[i, one])  # 将每个邻接矩阵的索引由大到小进行排列
            sele = one[oo[:pos_num]]  # 获得前pos_num个邻居节点
            pos[i, sele] = 1  # 将那些邻居节点标记为1，正样本
            k += 1
        else:
            pos[i, one] = 1  # 如果改行不够pos_num个节点，则全部标记为正样本

    pos = sp.coo_matrix(pos)
    sp.save_npz("../data/acm/acm_pos.npz", pos)
    # np.savetxt("../data/acm/acm_pos_4019.txt", pos)


def get_pos_dblp_npz(sc=0):
    path = "../data/dblp/"

    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")

    apa = normalize_adj(apa)
    apcpa = normalize_adj(apcpa)
    aptpa = normalize_adj(aptpa)

    pos_num = 1000
    p = 4057
    apa = apa / apa.sum(axis=-1).reshape(-1, 1)
    apcpa = apcpa / apcpa.sum(axis=-1).reshape(-1, 1)
    aptpa = aptpa / aptpa.sum(axis=-1).reshape(-1, 1)
    all = (apa + apcpa + aptpa).A.astype("float32")
    all_ = (all > 0).sum(-1)
    print(pos_num, all_.max(), all_.min(), all_.mean())  # 1000 4057 579 2206.4572344096623

    pos = np.zeros((p, p))
    k = 0
    for i in range(len(all)):
        one = all[i].nonzero()[0]  # 输出每一行不为零的数的索引（即节点i的邻居节点）
        if len(one) > pos_num:
            oo = np.argsort(-all[i, one])  # 将每个邻接矩阵的索引由大到小进行排列
            sele = one[oo[:pos_num]]  # 获得前pos_num个邻居节点
            pos[i, sele] = 1  # 将那些邻居节点标记为1，正样本
            k += 1
        else:
            pos[i, one] = 1  # 如果该行不够pos_num个节点，则全部标记为正样本

    pos = sp.coo_matrix(pos)
    sp.save_npz("../data/dblp/dblp_pos.npz", pos)
    # np.savetxt("../data/dblp/dblp_pos_npz.txt", pos)


def get_pos_freebase_npz(sc=0):
    path = "../data/freebase/"

    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")

    mam = normalize_adj(mam)
    mdm = normalize_adj(mdm)
    mwm = normalize_adj(mwm)

    pos_num = 80
    p = 3492
    mam = mam / mam.sum(axis=-1).reshape(-1, 1)
    mdm = mdm / mdm.sum(axis=-1).reshape(-1, 1)
    mwm = mwm / mwm.sum(axis=-1).reshape(-1, 1)
    all = (mam + mdm + mwm).A.astype("float32")
    all_ = (all > 0).sum(-1)
    print(pos_num, all_.max(), all_.min(), all_.mean())  # 80 451 1 74.75200458190149

    pos = np.zeros((p, p))
    k = 0
    for i in range(len(all)):
        one = all[i].nonzero()[0]  # 输出每一行不为零的数的索引（即节点i的邻居节点）
        if len(one) > pos_num:
            oo = np.argsort(-all[i, one])  # 将每个邻接矩阵的索引由大到小进行排列
            sele = one[oo[:pos_num]]  # 获得前pos_num个邻居节点
            pos[i, sele] = 1  # 将那些邻居节点标记为1，正样本
            k += 1
        else:
            pos[i, one] = 1  # 如果该行不够pos_num个节点，则全部标记为正样本

    pos = sp.coo_matrix(pos)
    sp.save_npz("../data/freebase/freebase_pos.npz", pos)
    # np.savetxt("../data/freebase/freebase_pos_npz.txt", pos)


def get_pos_aminer_npz(sc=0):
    path = "../data/aminer/"

    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")

    pap = normalize_adj(pap)
    prp = normalize_adj(prp)

    pos_num = 15
    p = 6564
    pap = pap / pap.sum(axis=-1).reshape(-1, 1)
    prp = prp / prp.sum(axis=-1).reshape(-1, 1)
    all = (pap + prp).A.astype("float32")
    all_ = (all > 0).sum(-1)
    print(pos_num, all_.max(), all_.min(), all_.mean())  # 15 216 1 21.583790371724557

    pos = np.zeros((p, p))
    k = 0
    for i in range(len(all)):
        one = all[i].nonzero()[0]  # 输出每一行不为零的数的索引（即节点i的邻居节点）
        if len(one) > pos_num:
            oo = np.argsort(-all[i, one])  # 将每个邻接矩阵的索引由大到小进行排列
            sele = one[oo[:pos_num]]  # 获得前pos_num个邻居节点
            pos[i, sele] = 1  # 将那些邻居节点标记为1，正样本
            k += 1
        else:
            pos[i, one] = 1  # 如果改行不够pos_num个节点，则全部标记为正样本

    pos = sp.coo_matrix(pos)
    sp.save_npz("../data/aminer/aminer_pos.npz", pos)
    # np.savetxt("../data/acm/acm_pos_4019.txt", pos)


def get_pos_amazon_pkl(sc=3):
    data = pkl.load(open("../data/amazon/amazon.pkl", "rb"))

    adj1 = data["IVI"] + np.eye(data["IVI"].shape[0])*sc
    adj2 = data["IBI"] + np.eye(data["IBI"].shape[0])*sc
    adj3 = data["IOI"] + np.eye(data["IOI"].shape[0])*sc

    adj1 = sp.coo_matrix(adj1)
    adj2 = sp.coo_matrix(adj2)
    adj3 = sp.coo_matrix(adj3)

    adj1 = normalize_adj(adj1)
    adj2 = normalize_adj(adj2)
    adj3 = normalize_adj(adj3)

    pos_num = 100
    p = 7621
    adj1 = adj1 / adj1.sum(axis=-1).reshape(-1, 1)
    adj2 = adj2 / adj2.sum(axis=-1).reshape(-1, 1)
    adj3 = adj3 / adj3.sum(axis=-1).reshape(-1, 1)
    all = (adj1 + adj2 + adj3).A.astype("float32")
    all_ = (all > 0).sum(-1)
    print(pos_num, all_.max(), all_.min(), all_.mean())  # 100 1210 1 163.2737173599265

    pos = np.zeros((p, p))
    k = 0
    for i in range(len(all)):
        one = all[i].nonzero()[0]  # 输出每一行不为零的数的索引（即节点i的邻居节点）
        if len(one) > pos_num:
            oo = np.argsort(-all[i, one])  # 将每个邻接矩阵的索引由大到小进行排列
            sele = one[oo[:pos_num]]  # 获得前pos_num个邻居节点
            pos[i, sele] = 1  # 将那些邻居节点标记为1，正样本
            k += 1
        else:
            pos[i, one] = 1  # 如果改行不够pos_num个节点，则全部标记为正样本

    pos = sp.coo_matrix(pos)
    sp.save_npz("../data/amazon/amazon_pos.npz", pos)


def get_pos_acm_mat(sc=0):
    data = sio.loadmat('../data/acm/acm.mat')

    pap = data["PAP"] + np.eye(data["PAP"].shape[0]) * sc
    plp = data["PLP"] + np.eye(data["PLP"].shape[0]) * sc

    pap = sp.coo_matrix(pap)
    plp = sp.coo_matrix(plp)

    pap = normalize_adj(pap)
    plp = normalize_adj(plp)

    pos_num = 5
    p = 7621
    pap = pap / pap.sum(axis=-1).reshape(-1, 1)
    plp = plp / plp.sum(axis=-1).reshape(-1, 1)
    all = (pap + plp).A.astype("float32")
    all_ = (all > 0).sum(-1)
    print(pos_num, all_.max(), all_.min(), all_.mean())  # 5 1247 1 734.4459504132232

    pos = np.zeros((p, p))
    k = 0
    for i in range(len(all)):
        one = all[i].nonzero()[0]  # 输出每一行不为零的数的索引（即节点i的邻居节点）
        if len(one) > pos_num:
            oo = np.argsort(-all[i, one])  # 将每个邻接矩阵的索引由大到小进行排列
            sele = one[oo[:pos_num]]  # 获得前pos_num个邻居节点
            pos[i, sele] = 1  # 将那些邻居节点标记为1，正样本
            k += 1
        else:
            pos[i, one] = 1  # 如果改行不够pos_num个节点，则全部标记为正样本

    np.savetxt("../data/acm/acm_pos_3025.txt", pos)


def get_pos_dblp_mat(sc=0):
    data = sio.loadmat('../data/dblp/dblp.mat')
    apa = data["net_APA"] + np.eye(data["net_APA"].shape[0]) * sc
    apcpa = data["net_APCPA"] + np.eye(data["net_APCPA"].shape[0]) * sc
    aptpa = data["net_APTPA"] + np.eye(data["net_APTPA"].shape[0]) * sc

    apa = sp.coo_matrix(apa)
    apcpa = sp.coo_matrix(apcpa)
    aptpa = sp.coo_matrix(aptpa)

    apa = normalize_adj(apa)
    apcpa = normalize_adj(apcpa)
    aptpa = normalize_adj(aptpa)

    # apa = apa.to_dense()
    # apcpa = apcpa.to_dense()
    # aptpa = aptpa.to_dense()

    pos_num = 500
    p = 4057
    apa = apa / apa.sum(axis=-1).reshape(-1, 1)
    apcpa = apcpa / apcpa.sum(axis=-1).reshape(-1, 1)
    aptpa = aptpa / aptpa.sum(axis=-1).reshape(-1, 1)
    all = (apa + apcpa + aptpa).A.astype("float32")
    all_ = (all > 0).sum(-1)
    print(all_.max(), all_.min(), all_.mean())   # 4057 458 2168.913236381563

    pos = np.zeros((p, p))
    k = 0
    for i in range(len(all)):
        one = all[i].nonzero()[0]  # 输出每一行不为零的数的索引（即节点i的邻居节点）
        if len(one) > pos_num:
            oo = np.argsort(-all[i, one])  # 将每个邻接矩阵的索引由大到小进行排列
            sele = one[oo[:pos_num]]  # 获得前pos_num个邻居节点
            pos[i, sele] = 1  # 将那些邻居节点标记为1，正样本
            k += 1
        else:
            pos[i, one] = 1  # 如果改行不够pos_num个节点，则全部标记为正样本

    np.savetxt("../data/dblp/dblp_pos_mat.txt", pos)


if __name__ == '__main__':
    # get_pos_acm_mat(0)
    # get_pos_dblp_mat(0)

    # get_pos_acm_npz(0)
    # get_pos_dblp_npz(0)
    # get_pos_freebase_npz(0)
    # get_pos_aminer_npz(0)
    get_pos_amazon_pkl(0)
