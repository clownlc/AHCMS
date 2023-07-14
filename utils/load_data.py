import random

import pickle as pkl
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder

from .knn import get_knn_graph
from .params import set_params


def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True


def new_graph(edge_index, weight, n, device):
    edge_index = edge_index.cpu().numpy()
    indices = th.from_numpy(
        np.vstack((edge_index[0], edge_index[1])).astype(np.int64)).to(device)
    values = weight
    shape = th.Size((n, n))
    return th.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_acm_npz(sc=3):
    # The order of node types: 0 p 1 a 2 s
    path = "data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_p = sp.load_npz(path + "p_feat.npz")

    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    pos = sp.load_npz(path + "acm_pos.npz")

    knn_adj = get_knn_graph(feat_p.A, 5)
    knn_adj = sp.coo_matrix(knn_adj)

    feat_p = th.FloatTensor(preprocess_features(feat_p))

    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    knn_adj = sparse_mx_to_torch_sparse_tensor(knn_adj)

    adj_list = [pap, psp]

    return feat_p, adj_list, pos, knn_adj, label


def load_dblp_npz(sc=3):
    # The order of node types: 0 p 1 a 2 s
    path = "data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")

    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    pos = sp.load_npz(path + "dblp_pos.npz")

    knn_adj = get_knn_graph(feat_a.A, 5)
    knn_adj = sp.coo_matrix(knn_adj)

    feat_a = th.FloatTensor(preprocess_features(feat_a))

    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    knn_adj = sparse_mx_to_torch_sparse_tensor(knn_adj)

    adj_list = [apa, apcpa, aptpa]

    return feat_a, adj_list, pos, knn_adj, label


def load_aminer_npz(sc=3):
    # The order of node types: 0 p 1 a 2 s
    path = "data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_p = sp.eye(label.shape[0])

    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "aminer_pos.npz")

    knn_adj = get_knn_graph(feat_p.A, 5)
    knn_adj = sp.coo_matrix(knn_adj)

    feat_p = th.FloatTensor(preprocess_features(feat_p))

    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    knn_adj = sparse_mx_to_torch_sparse_tensor(knn_adj)

    adj_list = [pap, prp]

    return feat_p, adj_list, pos, knn_adj, label


def load_freebase_npz(sc=0):
    # The order of node types: 0 p 1 a 2 s
    path = "data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_m = sp.eye(label.shape[0])

    mam = sp.load_npz(path + "mam.npz")
    mam = mam.todense()
    mam = mam + np.eye(mam.shape[0])*sc

    mdm = sp.load_npz(path + "mdm.npz")
    mdm = mdm.todense()
    mdm = mdm + np.eye(mdm.shape[0])*sc

    mwm = sp.load_npz(path + "mwm.npz")
    mwm = mwm.todense()
    mwm = mwm + np.eye(mwm.shape[0])*sc

    mam = sp.coo_matrix(mam)
    mdm = sp.coo_matrix(mdm)
    mwm = sp.coo_matrix(mwm)

    pos = sp.load_npz(path + "freebase_pos.npz")

    knn_adj = get_knn_graph(feat_m.A, 5)
    knn_adj = sp.coo_matrix(knn_adj)

    feat_m = th.FloatTensor(preprocess_features(feat_m))

    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    knn_adj = sparse_mx_to_torch_sparse_tensor(knn_adj)

    adj_list = [mam, mdm, mwm]

    return feat_m, adj_list, pos, knn_adj, label


def load_amazon_pkl(sc=3):
    data = pkl.load(open("data/amazon/amazon.pkl", "rb"))
    label = data['label']

    adj1 = data["IVI"] + np.eye(data["IVI"].shape[0])*sc
    adj2 = data["IBI"] + np.eye(data["IBI"].shape[0])*sc
    adj3 = data["IOI"] + np.eye(data["IOI"].shape[0])*sc

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    pos = sp.load_npz("data/amazon/amazon_pos.npz")

    knn_adj = get_knn_graph(truefeatures.A, 5)
    knn_adj = sp.coo_matrix(knn_adj)

    truefeatures = th.FloatTensor(preprocess_features(truefeatures))

    adj1 = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj1))
    adj2 = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj2))
    adj3 = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj3))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    knn_adj = sparse_mx_to_torch_sparse_tensor(knn_adj)

    adj_list = [adj1, adj2, adj3]

    return truefeatures, adj_list, pos, knn_adj, label


def load_acm_mat(sc=3):
    args = set_params()
    data = sio.loadmat('data/acm/acm.mat')
    label = data['label']
    features = data['feature'].astype(float)

    # pca = PCA(n_components=args.n_components)
    # feats = pca.fit_transform(features)
    feats = features

    pap = data["PAP"] + np.eye(data["PAP"].shape[0]) * sc
    plp = data["PLP"] + np.eye(data["PLP"].shape[0]) * sc
    pos = np.loadtxt('data/acm/acm_pos_3025.txt')
    knn_adj = get_knn_graph(features, 5)

    pap = sp.coo_matrix(pap)
    plp = sp.coo_matrix(plp)
    pos = sp.coo_matrix(pos)
    knn_adj = sp.coo_matrix(knn_adj)
    feats = sp.csr_matrix(feats)

    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    plp = sparse_mx_to_torch_sparse_tensor(normalize_adj(plp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    knn_adj = sparse_mx_to_torch_sparse_tensor(knn_adj)
    feats = th.FloatTensor(preprocess_features(feats))

    adj_list = [pap, plp]

    return feats, adj_list, pos, knn_adj, label


def load_dblp_mat(sc=3):
    args = set_params()
    data = sio.loadmat('data/dblp/dblp.mat')
    label = data['label']
    features = data['features'].astype(float)

    # pca = PCA(n_components=args.n_components)
    # feats = pca.fit_transform(features)
    feats = features

    apa = data["net_APA"] + np.eye(data["net_APA"].shape[0]) * sc
    apcpa = data["net_APCPA"] + np.eye(data["net_APCPA"].shape[0]) * sc
    aptpa = data["net_APTPA"] + np.eye(data["net_APTPA"].shape[0]) * sc
    pos = np.loadtxt('data/dblp/dblp_pos_mat.txt')
    knn_adj = get_knn_graph(features, 5)

    apa = sp.coo_matrix(apa)
    apcpa = sp.coo_matrix(apcpa)
    aptpa = sp.coo_matrix(aptpa)
    pos = sp.coo_matrix(pos)
    knn_adj = sp.coo_matrix(knn_adj)
    feats = sp.csr_matrix(feats)

    # row = apcpa.row
    # col = apcpa.col
    # with open(f"data/dblp/dblp_graph_apcpa.txt", 'w+') as f:
    #     for i in range(len(row)):
    #         if row[i] != col[i]:
    #             temp = f"{row[i]} {col[i]}"
    #             f.write(temp)
    #             f.write('\n')
    #             f.flush()
    #     f.close()
    # print(f"完成了dblp_pos_graph.txt的修改！！")

    # label = th.FloatTensor(label)
    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    knn_adj = sparse_mx_to_torch_sparse_tensor(knn_adj)
    feats = th.FloatTensor(preprocess_features(feats))

    adj_list = [apa, apcpa, aptpa]

    # edge_index_apa = np.genfromtxt("data/dblp/dblp_graph_apa.txt", dtype=np.int32)
    # edge_index_apcpa = np.genfromtxt("data/dblp/dblp_graph_apcpa.txt", dtype=np.int32)
    # edge_index_aptpa = np.genfromtxt("data/dblp/dblp_graph_aptpa.txt", dtype=np.int32)
    # edge_index_apa = edge_index_apa.transpose()
    # edge_index_apcpa = edge_index_apcpa.transpose()
    # edge_index_aptpa = edge_index_aptpa.transpose()
    #
    # edge_index_list = [edge_index_apa, edge_index_apcpa, edge_index_aptpa]

    return feats, adj_list, pos, knn_adj, label


def load_aminer_mat(sc=3):
    data = sio.loadmat('../data/dblp/dblp.mat')
    label = data['label']

    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0]) * sc
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0]) * sc

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test


def load_freebase_mat(sc=3):
    data = sio.loadmat('../data/dblp/dblp.mat')
    label = data['label']

    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0]) * sc
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0]) * sc

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test


def load_data_mat(dataset, sc):
    if dataset == "acm":
        data = load_acm_mat(sc)
    elif dataset == "dblp":
        data = load_dblp_mat(sc)
    elif dataset == "aminer":
        data = load_aminer_mat(sc)
    elif dataset == "freebase":
        data = load_freebase_mat(sc)
    return data


def load_data_npz(dataset, sc):
    if dataset == "acm":
        data = load_acm_npz(sc)
    elif dataset == "dblp":
        data = load_dblp_npz(sc)
    elif dataset == "aminer":
        data = load_aminer_npz(sc)
    elif dataset == "freebase":
        data = load_freebase_npz(sc)
    elif dataset == "amazon":
        data = load_amazon_pkl(sc)
    return data
