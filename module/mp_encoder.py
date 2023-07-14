import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.Tanh()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, active):
        seq_fts = self.fc(seq)
        if active:
            seq_fts = self.act(seq_fts)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return out


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        # print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i]
        return z_mp


class SemanticAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class FusionLayer(nn.Module):

    def __init__(self, args):
        super(FusionLayer, self).__init__()
        act_func = 'relu'
        views = args.views
        use_bn = True  # Whether add BatchNormalize into fusion module.
        mlp_layers = 1  # The number of layer of fusion module.
        in_features = args.hidden_dim
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'tanh':
            self.act = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('Activate function must be ReLU or Tanh.')
        self.layers = [self._make_layers(in_features * views, in_features, self.act, use_bn)]
        if mlp_layers > 1:
            layers = [self._make_layers(in_features, in_features,
                                        self.act if _ < (mlp_layers - 2) else nn.Identity(),
                                        use_bn if _ < (mlp_layers - 2) else False) for _ in range(mlp_layers - 1)]
            self.layers += layers
        self.layers = nn.Sequential(*self.layers)

    def forward(self, h):
        h = torch.cat(h, dim=-1)
        z = self.layers(h)
        return z

    def _make_layers(self, in_features, out_features, act, bn=False):
        layers = nn.ModuleList()
        layers.append(nn.Linear(in_features, out_features))
        layers.append(act)
        if bn:
            layers.append(nn.BatchNorm1d(out_features))
        return nn.Sequential(*layers)


class Mp_encoder(nn.Module):
    def __init__(self, P, feats_dim, hidden_dim, attn_drop, args):
        super(Mp_encoder, self).__init__()
        self.P = P
        self.node_level = nn.ModuleList([GCN(feats_dim, hidden_dim) for _ in range(P)])
        self.att = Attention(hidden_dim, attn_drop)
        self.semantic_attention = SemanticAttention(hidden_dim)

        self.fusion_layer = FusionLayer(args)

    def forward(self, h, mps):
        embeds = []
        for i in range(len(mps)):
            embeds.append(self.node_level[i](h, mps[i], active=True))

        # semantic_embeddings = torch.stack(embeds, dim=1)
        z_mp = self.att(embeds)
        # z_mp = self.semantic_attention(semantic_embeddings)

        # z_mp = self.fusion_layer(embeds)

        return z_mp, embeds
