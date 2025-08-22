import torch
import torch.nn as nn
import torch.nn.functional as F


class Tcn(nn.Module):
    def __init__(self, c_in, c_out):
        super(Tcn, self).__init__()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.filter_convs=torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 2), padding=(0, 0), stride=(1, 1), bias=True)
        self.gate_convs=torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 2), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self,x):
        x = nn.functional.pad(x, (1, 0, 0, 0))
        filter = self.filter_convs(x)
        filter = torch.tanh(filter)
        gate = self.gate_convs(x)
        gate = torch.sigmoid(gate)
        x = filter * gate
        return x

class temporal_graph_generator(nn.Module):
    def __init__(self, num_nodes, num_steps, device, N_t, temporal_dim):
        super(temporal_graph_generator, self).__init__()
        self.nodetime = nn.Parameter(torch.randn(N_t, temporal_dim[0]).to(device),
                                    requires_grad=True).to(device)
        self.nodenum = nn.Parameter(torch.randn(num_nodes, temporal_dim[1]).to(device),
                                     requires_grad=True).to(device)
        self.timevec1 = nn.Parameter(torch.randn(num_steps, temporal_dim[2]).to(device),
                                     requires_grad=True).to(device)
        self.timevec2 = nn.Parameter(torch.randn(num_steps, temporal_dim[3]).to(device),
                                     requires_grad=True).to(device)
        self.k = nn.Parameter(torch.randn(temporal_dim[0], temporal_dim[1], temporal_dim[2], temporal_dim[3]).to(device),
                              requires_grad=True).to(device)
    def forward(self, t):
        nodetime = self.nodetime[t]
        nodenum = self.nodenum
        timevec1 = self.timevec1
        timevec2 = self.timevec2
        k = self.k
        adp1 = torch.einsum("ad, defg->aefg", [nodetime, k])
        adp2 = torch.einsum("he, aefg->ahfg", [nodenum, adp1])
        adp3 = torch.einsum("bf, ahfg->ahbg", [timevec1, adp2])
        adp4 = torch.einsum("cg, ahbg->ahbc", [timevec2, adp3])
        adp = F.softmax(F.relu(adp4), dim=3)
        return adp

class temporal_block(nn.Module):
    def __init__(self, drop, num_nodes, num_steps, N_t, device, temporal_dim, hid_dim):
        super(temporal_block, self).__init__()
        self.drop = drop
        self.temporal_graph_generator = temporal_graph_generator(num_nodes, num_steps, device, N_t, temporal_dim)
        self.tcn = Tcn(hid_dim, hid_dim)

    def forward(self, x, t):
        x = self.tcn(x)
        adp = self.temporal_graph_generator(t)
        out = torch.einsum("ncvl, nvlw->ncvw", [x, adp])
        out = F.dropout(out, self.drop, training=self.training)
        return out
