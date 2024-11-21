
from LGTSNet.modules import *
from RetNet.src.retnet import RetNet
from LGTSNet.CA import CA_Block
from layers import GraphConvolution

class BaseModel(nn.Module):
    def __init__(self, input_shape=None, output_shape=None):
        super(BaseModel, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __build_pseudo_input(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
        temp_x_ = torch.rand(input_shape)
        temp_x = temp_x_.unsqueeze(0)
        return temp_x

    def get_tensor_shape(self, forward_func, input_shape=None):
        pseudo_x = self.__build_pseudo_input(input_shape)
        pseudo_y = forward_func(pseudo_x)
        return pseudo_y.shape



class LGTSNet(BaseModel):
    def __init__(self, input_shape=(14, 128)):
        super().__init__()
        chans: int = input_shape[0]
        expand_ch = chans * 4
        self.conv_expan_1 = nn.Sequential(
            nn.Conv1d(chans, expand_ch, 15, groups=chans),
        )
        self.dc_1 =nn.Sequential(nn.Conv1d(expand_ch, expand_ch, 15, groups=expand_ch),nn.BatchNorm1d(expand_ch),nn.ReLU(),nn.Dropout(0.5)
                )

        self.sc_1 = nn.Sequential(SC(expand_ch, expand_ch, kernel_size=15, padding=15 // 2),SC(expand_ch, expand_ch, kernel_size=15, padding=15 // 4),nn.BatchNorm1d(expand_ch),nn.ReLU(),nn.Dropout(0.5)
                )
        self.conv_expan_2 = nn.Sequential(
            nn.Conv1d(chans, expand_ch, 7, groups=chans),
        )
        self.dc_2 = nn.Sequential(nn.Conv1d(expand_ch, expand_ch, 7, groups=expand_ch),nn.BatchNorm1d(expand_ch),nn.ReLU(),nn.Dropout(0.5)
                )

        self.sc_2 = nn.Sequential(SC(expand_ch, expand_ch, kernel_size=7, padding=7 // 2),SC(expand_ch, expand_ch, kernel_size=7, padding=7 // 4),nn.BatchNorm1d(expand_ch),nn.ReLU(),nn.Dropout(0.5)
)
        self.conv_expan_3 = nn.Sequential(nn.Conv1d(chans, expand_ch, 3, groups=chans),)
        self.dc_3 = nn.Sequential(nn.Conv1d(expand_ch, expand_ch, 3, groups=expand_ch),nn.BatchNorm1d(expand_ch),nn.ReLU(),nn.Dropout(0.5))
        self.sc_3 = nn.Sequential(SC(expand_ch, expand_ch, kernel_size=3, padding=3 // 2),SC(expand_ch, expand_ch, kernel_size=3, padding=3 // 4),nn.BatchNorm1d(expand_ch),nn.ReLU(),nn.Dropout(0.5))
        self.fusion_1 = nn.Sequential(nn.AvgPool1d(4))
        self.fusion_2 = nn.Sequential(nn.AvgPool1d(5))
        self.CA = CA_Block(channel=56, h=1, w=14, reduction=2)
        self.retnet = RetNet(4, 56, 128, 7, double_v_dim=True)
        self.fc = nn.Sequential(nn.Linear(784, 4))
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(56, 14),requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, 56, 1), dtype=torch.float32),requires_grad=True)
        self.local_filter_weight_v1 = nn.Parameter(torch.FloatTensor(56, 14),requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight_v1)
        self.local_filter_bias_v1 = nn.Parameter(torch.zeros((1, 56, 1), dtype=torch.float32),requires_grad=True)
        self.global_adj = nn.Parameter(torch.FloatTensor(56, 56), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(56)
        self.bn_ = nn.BatchNorm1d(56)
        # learn the global network of networks
        self.GCN = GraphConvolution(14, 14)


    # @timer_wrap
    def forward(self, x: torch.Tensor):
        x = torch.squeeze(x, dim=1)
        seq_embed = self.forward_embed(x)
        gcn_input = seq_embed
        adj = self.get_adj(gcn_input)
        gcn_input = self.bn(gcn_input)
        gcn_input = self.GCN(gcn_input, adj)
        gcn_out = self.bn_(gcn_input)
        retnet_input = seq_embed
        retnet_input = retnet_input.permute(0, 2, 1)
        batch= retnet_input.shape[0]
        ## (b, w, c)
        s_n_1s = [
            [
                torch.zeros(56 // 7, self.retnet.v_dim // 7).unsqueeze(0).repeat(batch, 1, 1)
                for _ in range(7)
            ]
            for _ in range(4)
        ]
        Y_recurrent = []
        for i in range(14):
            y_n, s_ns = self.retnet.forward_recurrent(retnet_input[:, i:i + 1, :], s_n_1s, i)
            Y_recurrent.append(y_n)
            s_n_1s = s_ns

        retnet_output = torch.concat(Y_recurrent, dim=1)
        retnet_output = retnet_output.permute(0, 2, 1)
        out = retnet_output + gcn_out
        batch = out.shape[0]
        out = out.reshape(batch, -1)
        out = self.fc(out)

        return out

    def forward_embed(self, x):
        x_v0 = self.conv_expan_1(x)
        x_v0 = self.dc_1(x_v0)
        x_v0 = self.fusion_1(x_v0)
        x_v0 = self.sc_1(x_v0)
        x_v1 = self.conv_expan_2(x)
        x_v1 = self.dc_2(x_v1)
        x_v1 = self.fusion_1(x_v1)
        x_v1 = self.sc_2(x_v1)
        x_v2 = self.conv_expan_3(x)
        x_v2 = self.dc_2(x_v2)
        x_v2 = self.fusion_1(x_v2)
        x_v2 = self.sc_3(x_v2)
        x_fuse = torch.cat((x_v0, x_v1, x_v2), dim=-1)
        out = self.fusion_2(x_fuse)
        out = torch.unsqueeze(out, dim=2)
        out = self.CA(out)
        out = torch.squeeze(out, dim=2)
        return out


    def get_adj(self, x, self_loop=True):
        # x: b, node, feature
        adj = self.self_similarity(x)  # b, n, n
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to('cuda:0')
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj


    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s
