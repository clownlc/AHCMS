import argparse
import sys

argv = sys.argv
# dataset = argv[1]
dataset = 'acm'


def acm_params():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--save_emb', action="store_true")
    # parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--sc', type=int, default=0)  # ----
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=200)
    # parser.add_argument('--n_components', type=int, default=50)
    parser.add_argument('--pos_num', type=int, default=5)
    parser.add_argument('--order', type=int, default=2, help='to compute order-th power of adj')
    parser.add_argument('--parm_kl', type=float, default=1)
    # parser.add_argument('--influence', default=False, action='store_true', help='Use Inluence contrastive')

    parser.add_argument('--parm_contr', type=float, default=1)
    parser.add_argument('--parm_nContrast', type=float, default=1)
    parser.add_argument('--parm_corr', type=float, default=1)
    parser.add_argument('--parm_Ncontr', type=float, default=1, help='temperature for Ncontrast loss')

    # The parameters of evaluation
    # parser.add_argument('--reg_lambda', type=float, default=1, help='View Learner Edge Perturb Regularization Strength')

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)  # 0.005
    # parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()

    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--sc', type=int, default=0)  # ----
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=64)  # 128 64
    parser.add_argument('--nb_epochs', type=int, default=500)
    parser.add_argument('--n_components', type=int, default=50)
    parser.add_argument('--pos_num', type=int, default=1000)
    parser.add_argument('--order', type=int, default=2, help='to compute order-th power of adj')
    parser.add_argument('--parm_kl', type=float, default=1)
    parser.add_argument('--influence', default=False, action='store_true', help='Use Inluence contrastive')

    parser.add_argument('--parm_Ncontr', type=float, default=1, help='temperature for Ncontrast loss')
    parser.add_argument('--parm_nContrast', type=float, default=0.1)  # 0.1
    parser.add_argument('--parm_contr', type=float, default=1)
    parser.add_argument('--parm_corr', type=float, default=0.1)  # 0.1

    # The parameters of evaluation
    parser.add_argument('--reg_lambda', type=float, default=1, help='View Learner Edge Perturb Regularization Strength')

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)  # 0.001
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()

    return args


def aminer_params():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--save_emb', action="store_true")
    # parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="aminer")
    parser.add_argument('--sc', type=int, default=0)  # ----
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=500)
    # parser.add_argument('--n_components', type=int, default=50)
    parser.add_argument('--pos_num', type=int, default=5)
    parser.add_argument('--order', type=int, default=2, help='to compute order-th power of adj')
    parser.add_argument('--parm_kl', type=float, default=1)

    # parser.add_argument('--influence', default=False, action='store_true', help='Use Inluence contrastive')

    parser.add_argument('--parm_contr', type=float, default=0.1)  # 1 and 0.1
    parser.add_argument('--parm_nContrast', type=float, default=1)  # 10 1
    parser.add_argument('--parm_corr', type=float, default=10)  # 1 10
    parser.add_argument('--parm_Ncontr', type=float, default=0.0001, help='temperature for Ncontrast loss')  # 0.001 0.0001

    # The parameters of evaluation
    # parser.add_argument('--reg_lambda', type=float, default=1, help='View Learner Edge Perturb Regularization Strength')

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=100)  # 10
    parser.add_argument('--lr', type=float, default=0.005)  # 0.008 稳定
    # parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()

    return args


def freebase_params():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--save_emb', action="store_true")
    # parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--views', type=int, default=2)
    parser.add_argument('--sc', type=int, default=0)  # 10-20：模型稳定
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=300)
    # parser.add_argument('--n_components', type=int, default=50)
    parser.add_argument('--pos_num', type=int, default=845)
    parser.add_argument('--order', type=int, default=2, help='to compute order-th power of adj')
    parser.add_argument('--parm_kl', type=float, default=1)
    # parser.add_argument('--influence', default=False, action='store_true', help='Use Inluence contrastive')

    parser.add_argument('--parm_contr', type=float, default=0.1)  # 1 0.1
    parser.add_argument('--parm_nContrast', type=float, default=1)  # 10 1
    parser.add_argument('--parm_corr', type=float, default=10)  # 1 10 ----
    parser.add_argument('--parm_Ncontr', type=float, default=0.0001, help='temperature for Ncontrast loss')  # 0.001 0.0001

    # The parameters of evaluation
    # parser.add_argument('--reg_lambda', type=float, default=1, help='View Learner Edge Perturb Regularization Strength')

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.008)  # 0.005
    # parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()

    return args


def amazon_params():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--save_emb', action="store_true")
    # parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="amazon")
    parser.add_argument('--sc', type=int, default=0)  # ----
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=150)
    # parser.add_argument('--n_components', type=int, default=50)
    parser.add_argument('--pos_num', type=int, default=5)
    parser.add_argument('--order', type=int, default=2, help='to compute order-th power of adj')
    parser.add_argument('--parm_kl', type=float, default=1)
    # parser.add_argument('--influence', default=False, action='store_true', help='Use Inluence contrastive')

    parser.add_argument('--parm_Ncontr', type=float, default=1, help='temperature for Ncontrast loss')
    parser.add_argument('--parm_nContrast', type=float, default=1)
    parser.add_argument('--parm_contr', type=float, default=1)
    parser.add_argument('--parm_corr', type=float, default=1)

    # The parameters of evaluation
    # parser.add_argument('--reg_lambda', type=float, default=1, help='View Learner Edge Perturb Regularization Strength')

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.005)  # 0.005
    # parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()

    return args


def set_params(dataset):
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "aminer":
        args = aminer_params()
    elif dataset == "freebase":
        args = freebase_params()
    elif dataset == "amazon":
        args = amazon_params()
    return args
