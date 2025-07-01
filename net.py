from math import ceil
from attention_model6 import *
import argparse

from layer import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GNNStack(nn.Module):
    """ The stack layers of GNN.

    """

    def __init__(self, gnn_model_type, num_layers, groups, pool_ratio, kern_size,
                 in_dim, hidden_dim, out_dim, 
                 seq_len, num_nodes, fft_channel,stgere ,num_classes,fft, relu2, mtpool, dropout=0.1, activation=nn.ReLU()):

        super().__init__()
        
        # TODO: Sparsity Analysis
        self.fft_channel=fft_channel
        self.mtpool=mtpool
        k_neighs = self.num_nodes = num_nodes
        self.fft=fft
        self.relu2=relu2
        nodes_len= int(2500 / groups)
        nodes_pro=nodes_len * 4
        self.ffn1 = nn.Sequential(
            nn.Linear(nodes_len, nodes_pro),
            nn.BatchNorm1d(nodes_pro),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(nodes_pro, nodes_len),
            nn.BatchNorm1d(nodes_len),
            nn.Dropout(p=0.5)
        )
        self.enc_embedding = DataEmbedding_wo_pos(12, fft_channel,0.1)
        self.num_graphs = groups
        self.encoder = Encoder(
            [
                EncoderLayer(
                    bandfreAttention(fft_channel, 1280, 1280, 2,
                                    8),
                    fft_channel, 64, 0.1, 'relu'
                ) for i in range(3)
            ]
        )
        self.num_feats = seq_len
        if seq_len % groups:
            self.num_feats += ( groups - seq_len % groups )
        self.g_constr = multi_shallow_embedding(num_nodes, k_neighs, groups, fft_channel,stgere)
        
        gnn_model, heads = self.build_gnn_model(gnn_model_type)
        
        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'
        paddings = [ (k - 1) // 2 for k in kern_size ]
        
        self.tconvs = nn.ModuleList(
            [nn.Conv2d(1, in_dim, (1, kern_size[0]), padding=(0, paddings[0]))] + 
            [nn.Conv2d(heads * in_dim, hidden_dim, (1, kern_size[layer+1]), padding=(0, paddings[layer+1])) for layer in range(num_layers - 2)] + 
            [nn.Conv2d(heads * hidden_dim, out_dim, (1, kern_size[-1]), padding=(0, paddings[-1]))]
        )
        
        self.gconvs = nn.ModuleList(
            [gnn_model(in_dim, heads * in_dim, groups)] + 
            [gnn_model(hidden_dim, heads * hidden_dim, groups) for _ in range(num_layers - 2)] + 
            [gnn_model(out_dim, heads * out_dim, groups)]
        )
        
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(heads * in_dim)] + 
            [nn.BatchNorm2d(heads * hidden_dim) for _ in range(num_layers - 2)] + 
            [nn.BatchNorm2d(heads * out_dim)]
        )
        
        self.left_num_nodes = []
        for layer in range(num_layers + 1):
            left_node = round( num_nodes * (1 - (pool_ratio*layer)) )
            if left_node > 0:
                self.left_num_nodes.append(left_node)
            else:
                self.left_num_nodes.append(1)
        self.diffpool = nn.ModuleList(
            [Dense_TimeDiffPool2d(self.left_num_nodes[layer], self.left_num_nodes[layer+1], kern_size[layer], paddings[layer]) for layer in range(num_layers - 1)] +
            [Dense_TimeDiffPool2d(self.left_num_nodes[-2], self.left_num_nodes[-1], kern_size[-1], paddings[-1])]
        )

        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        
        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.linear = nn.Linear(heads * out_dim, num_classes)
        
        self.reset_parameters()
        adaptive_pooling_layers = []
        # ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=self.num_nodes // 3, Dim_output=self.hid, use_cuda=self.use_cuda)
        # adaptive_pooling_layers.append(ap)
        # ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=self.num_nodes//2, Dim_output=self.hid, use_cuda=self.use_cuda)
        # adaptive_pooling_layers.append(ap)
        ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=256, N_output=1, Dim_output=256,
                                    use_cuda=1)
        adaptive_pooling_layers.append(ap)
        # ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid4, N_output=1, Dim_output=self.hid4)
        # adaptive_pooling_layers.append(ap)
        # D = self.num_nodes//4

        # reduce_factor = 4
        # while D > 1:
        #     D = D // reduce_factor
        #     if D < 1:
        #         D = 1
        #     ap = Adaptive_Pooling_Layer(Heads=4, Dim_input=self.hid, N_output=D, Dim_output=self.hid4)
        #     adaptive_pooling_layers.append(ap)

        self.ap = nn.ModuleList(adaptive_pooling_layers)
        
    def reset_parameters(self):
        for tconv, gconv, bn, pool in zip(self.tconvs, self.gconvs, self.bns, self.diffpool):
            tconv.reset_parameters()
            gconv.reset_parameters()
            bn.reset_parameters()
            pool.reset_parameters()
        
        self.linear.reset_parameters()
        
        
    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1
        

    def forward(self, inputs: Tensor):
        
        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            x = F.pad(inputs, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            x = inputs#(8,1,12,2500)

        # x = x.squeeze(1)
        # # x = x.contiguous()
        # xffn = x.view(-1, x.size(2))
        #
        # # 应用 ffn1 到每个样本的每个导联
        # xffn = self.ffn1(xffn)
        #
        # # 将输出重塑回原始形状 (batch_size, channels, num_leads, signal_length)
        # x = xffn.view(x.size(0), 1, x.size(1), x.size(2))
        if self.fft ==1 :
            x = x.squeeze(1)
            enc_out = self.enc_embedding(x)#need (64,2,1280)
            x = self.encoder(enc_out)
            # x =Model(x)  # (64,1280,16)
            tensor_unsqueezed = x.unsqueeze(1)
            # 然后使用 transpose 或 permute 重新排列维度，得到 (8, 1, 24, 2500)
            x = tensor_unsqueezed.transpose(2, 3)
        else:
            x=x

        adj = self.g_constr(x.device)#(2,24,24)




        for tconv, gconv, bn, pool in zip(self.tconvs, self.gconvs, self.bns, self.diffpool):
            
            # x, adj = pool( gconv( tconv(x), adj ), adj )
            time_conv_output = tconv(x) #(B, C, N, F)

            # 然后将时间卷积的输出和邻接矩阵传入图卷积层
            graph_conv_input = (time_conv_output, adj)
            graph_conv_output = gconv(*graph_conv_input)  # 使用 * 来解包元组

            # 接下来，使用池化层处理图卷积的输出和当前的邻接矩阵
            x,adj = (graph_conv_output, adj)
            pooling_input = (graph_conv_output, adj)
            x, adj = pool(*pooling_input)
            # xmt, adjmt=x, adj
            x = self.activation( bn(x) ) #RELU
            # # #
            x = F.dropout(x, p=self.dropout, training=self.training)

        #
        # #mtpool
        if self.mtpool==1:
            A = adj  # (180,24,24)
            for layer in self.ap:  #
                x = layer(x, A)
            out=x
            # out = self.global_pool(x)#(180,256,24,52)
            out = out.view(out.size(0), -1)#(64,64)
            out = self.linear(out)#(64,6)
        else:
            # 应用全局平均池化，将特征图的每个通道的维度缩减为 1。
            out = self.global_pool(x)
            # 展平特征图，准备输入到全连接层。
            out = out.view(out.size(0), -1)
            # 通过全连接层进行分类。
            out = self.linear(out)

        return out

        # self.enc_embedding = DataEmbedding_wo_pos(enc_in=2, d_model=24, dropout=0.1)
        # self.num_graphs = groups
        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             bandfreAttention(d_model=24, seq_len_q=1280, seq_len_kv=1280, modes=self.modes,
        #                              n_heads=8),
        #             d_model=24, d_ff=64, dropout=dropout, activation='relu'
        #         ) for i in range(e_layers=3)
        #     ]
        # )