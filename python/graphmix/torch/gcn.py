import torch
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features, activation=None, dropout=0):
        super().__init__()
        self.l = torch.nn.Linear(in_features, out_features, bias=True)
        torch.nn.init.xavier_uniform_(self.l.weight)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, edge_norm):
        if self.dropout > 0:
            x = F.dropout(x, self.dropout)
        x = self.l(x)
        x = torch.sparse.mm(edge_norm, x)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation is not None:
            raise NotImplementedError
        return x

class SageConv(torch.nn.Module):
    def __init__(self, in_features, out_features, activation=None, dropout=0):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)
        self.linear_local = torch.nn.Linear(in_features, out_features, bias=True)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.linear_local.weight)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, edge_norm):
        if self.dropout > 0:
            feat = F.dropout(x, self.dropout)
        else:
            feat = x
        feat_trans_local = self.linear_local(feat)
        feat_trans_neigh = self.linear(torch.sparse.mm(edge_norm, feat))
        x = torch.cat([feat_trans_local, feat_trans_neigh], dim=1)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation is not None:
            raise NotImplementedError
        return x
