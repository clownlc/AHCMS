import torch
from torch import nn
from torch.nn import Sequential, Linear
from utils import set_params

# args = set_params()


class Generator(nn.Module):
    def __init__(self, encoder):
        super(Generator, self).__init__()

        self.encoder = encoder
        self.input_dim = args.n_components

        self.mlp_edge_model = Sequential(
            Linear(self.input_dim * 2, 1),
            #	ReLU(),
            #	Linear(mlp_edge_model_dim, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, feats, adj_list, edge_index_list):

        z_mp, embeds_list = self.encoder(feats, adj_list)
        edge_logits_list = list()

        for i in range(len(adj_list)):
            src, dst = edge_index_list[i][0], edge_index_list[i][1]  # src：边的序号；dst：与该边直接相连的序号
            emb_src = embeds_list[i][src]
            emb_dst = embeds_list[i][dst]
            edge_emb = torch.cat([emb_src, emb_dst], 1)
            edge_logits = self.mlp_edge_model(edge_emb)
            edge_logits_list.append(edge_logits)

        return edge_logits_list
