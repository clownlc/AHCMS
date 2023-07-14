from .evaluate import eva
from .load_data import load_data_mat, load_data_npz, new_graph, normalize, setup_seed, normalize_adj, sparse_mx_to_torch_sparse_tensor
from .params import set_params
from .logreg import LogReg
from .knn import get_knn_graph
from .ncontrast import get_A_r_flex, get_feature_dis, Ncontrast
from .pos import get_pos_freebase_npz
