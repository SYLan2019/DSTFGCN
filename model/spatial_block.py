import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class Dif_cov(nn.Module):
    def __init__(self, adj, device):
        super(Dif_cov, self).__init__()
        self.device = device
        self.adj_one = self.normalization(adj)
        
    def normalization(self, adjacency):
        identity = torch.diag_embed(torch.ones_like(adjacency.diag()))
        adjacency = adjacency + identity
        degree = torch.diag(adjacency.sum(dim=1))
        d_diffusion = 1 / degree
        d_diffusion[[d_diffusion == float('inf')]] = 0.
        L = d_diffusion @ adjacency
        return L

    def forward(self, x):
        x = torch.einsum('ncvl,vw->ncwl', (x, self.adj_one))
        return x

class Graph_Generator(nn.Module):
    def __init__(self, c_in, c_out, drop):
        super(Graph_Generator, self).__init__()
        self.drop = drop
        self.w = nn.Conv2d(c_in, c_out, 1)
        self.norm = 1 / sqrt(c_in)
        self.avg = nn.AvgPool2d(kernel_size=(c_out, 1), stride=1)

    def forward(self, x):
        l = self.w(x)
        l = F.softmax(l, dim=3)
        x1 = torch.einsum('ncvl,nqvl->ncqv', x, l)
        q_n = self.avg(x1).squeeze(2)
        attention = torch.matmul(q_n.transpose(1, 2), q_n) * self.norm
        adp = F.softmax(F.relu(attention), dim=2)
        return adp

class spatial_block(nn.Module):
    def __init__(self, drop, hid_dim, emb_dim, adj, device):
        super(spatial_block, self).__init__()
        self.drop = drop
        self.dif_cov = Dif_cov(adj, device)
        self.spatial_graph = Graph_Generator(hid_dim, emb_dim, drop)
        
    def forward(self, x):
        x = self.dif_cov(x)
        adp = self.spatial_graph(x)
        out = torch.einsum('ncvl,nvw->ncwl', (x, adp))
        out = F.dropout(out, self.drop, training=self.training)
        return out
