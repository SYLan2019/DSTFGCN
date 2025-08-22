from model.temporal_block import *
from model.spatial_block import *
class ST_block(nn.Module):
    def __init__(self, drop, num_nodes, num_steps, embed_dim, N_t, adj, temporal_dim, hid_dim, device):
        super(ST_block, self).__init__()
        self.spatial_block = spatial_block(drop, hid_dim,
                                           embed_dim, adj, device)
        self.temporal_block = temporal_block(drop, num_nodes, num_steps, N_t, device, temporal_dim, hid_dim)

        self.linear = torch.nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x, t):
        x_f = self.linear(x)
        x_s = self.spatial_block(x)
        x_t = self.temporal_block(x, t)
        x = torch.cat([x_f, x_s, x_t], dim=1)
        return x
class DSTFGCN(nn.Module):
    def __init__(self, device, config, adj, skip_channels,):
        super(DSTFGCN, self).__init__()
        self.drop = config['drop']
        self.layers = config['layers']
        self.num_nodes = config['num_nodes']
        self.hid_dim = config['hidden_dimension']
        self.residual_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.result_fuse = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.adj = torch.from_numpy(adj).to(device)
        self.ST_block = ST_block(self.drop, config['num_nodes'], 12, config['embedding_dimension'],
                                 config['N_t'], self.adj, config['temporal_dim'], self.hid_dim, device)
        self.layer_ST_block = nn.ModuleList()

        for layers in range(self.layers):
            self.layer_ST_block.append(ST_block(self.drop, config['num_nodes'], 12, config['embedding_dimension'],
                                 config['N_t'], self.adj, config['temporal_dim'], self.hid_dim, device))
            self.residual_convs.append(nn.Conv2d(in_channels=self.hid_dim,
                                                 out_channels=self.hid_dim,
                                                 kernel_size=(1, 1)))

            self.skip_convs.append(nn.Conv2d(in_channels=self.hid_dim,
                                             out_channels=skip_channels,
                                             kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(self.hid_dim))

            self.result_fuse.append(torch.nn.Conv2d(self.hid_dim * 6, self.hid_dim, kernel_size=(1, 1), padding=(0, 0),
                                stride=(1, 1), bias=True))

        self.start_conv = nn.Conv2d(in_channels=config['input_dim'],
                                    out_channels=self.hid_dim,
                                    kernel_size=(1, 1))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=self.hid_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=config['output_dim'] * self.hid_dim,
                                    out_channels=config['output_dim'],
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, input, t):
        x = input
        x = self.start_conv(x)
        skip = 0
        for i in range(self.layers):
            residual = x
            x_shar = self.ST_block(x, t)
            x_ind = self.layer_ST_block[i](x, t)
            x = torch.cat([x_shar, x_ind], dim=1)
            x = F.relu(x)
            x = self.result_fuse[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = torch.transpose(x, 3, 2)
        x = torch.reshape(x, (x.size(0), x.size(1) * x.size(2), x.size(3), 1))
        x = self.end_conv_2(x)
        return x
