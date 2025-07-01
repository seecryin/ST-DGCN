import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter


class multi_shallow_embedding(nn.Module):
    
    def __init__(self, num_nodes, k_neighs, num_graphs, fft_channel,stgere):
        super().__init__()
        self.stgere=stgere
        self.num_nodes = num_nodes
        self.k = k_neighs
        self.num_graphs = num_graphs

        self.emb_s = Parameter(Tensor(num_graphs, fft_channel, 1))
        self.emb_t = Parameter(Tensor(num_graphs, 1, fft_channel))
        
    def reset_parameters(self):
        init.xavier_uniform_(self.emb_s)
        init.xavier_uniform_(self.emb_t)


        
    def forward(self, device):
        
        # adj: [G, N, N]
        adj = torch.matmul(self.emb_s, self.emb_t).to(device)
        adj_flat = torch.zeros_like(adj).clone()
        if self.stgere == 1:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj=adj_flat
        if self.stgere == 2:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 3:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 4:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 5:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 6:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 7:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 8:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 9:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 10:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 11:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 12:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 13:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 14:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat
        if self.stgere == 15:
            # 步骤2：定义目标矩阵
            target_matrix = torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
            ])

            # 步骤3：赋值
            for i in range(self.num_graphs):
                adj_flat[i] = target_matrix

            adj = adj_flat

        else:
            adj = torch.matmul(self.emb_s, self.emb_t).to(device)
            adj = adj.clone()
            idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
            adj[:, idx, idx] = float('-inf')

            # top-k-edge adj
            adj_flat = adj.reshape(self.num_graphs, -1)
            indices = adj_flat.topk(k=self.k)[1].reshape(-1)

            idx = torch.tensor([i // self.k for i in range(indices.size(0))], device=device)

            adj_flat = torch.zeros_like(adj_flat).clone()
            adj_flat[idx, indices] = 1.
            adj = adj_flat.reshape_as(adj)
        return adj


class Group_Linear(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super().__init__()
                
        self.out_channels = out_channels
        self.groups = groups
        
        self.group_mlp = nn.Conv2d(in_channels * groups, out_channels * groups, kernel_size=(1, 1), groups=groups, bias=bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.group_mlp.reset_parameters()
        
        
    def forward(self, x: Tensor, is_reshape: False):
        """
        Args:
            x (Tensor): [B, C, N, F] (if not is_reshape), [B, C, G, N, F//G] (if is_reshape)
        """
        B = x.size(0)
        C = x.size(1)
        N = x.size(-2)
        G = self.groups
        
        if not is_reshape:
            # x: [B, C_in, G, N, F//G]
            x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        # x: [B, G*C_in, N, F//G]
        x = x.transpose(1, 2).reshape(B, G*C, N, -1)
        
        out = self.group_mlp(x)
        out = out.reshape(B, G, self.out_channels, N, -1).transpose(1, 2)
        
        # out: [B, C_out, G, N, F//G]
        return out


class DenseGCNConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lin = Group_Linear(in_channels, out_channels, groups, bias=False)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        init.zeros_(self.bias)
        
    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[:, idx, idx] += 1
        
        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
        return adj
        
        
    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [B, G, N, N]
        """
        adj = self.norm(adj, add_loop).unsqueeze(1)

        # x: [B, C, G, N, F//G]
        x = self.lin(x, False)
        
        out = torch.matmul(adj, x)
        
        # out: [B, C, N, F]
        B, C, _, N, _ = out.size()
        out = out.transpose(2, 3).reshape(B, C, N, -1)
        
        if self.bias is not None:
            out = out.transpose(1, -1) + self.bias
            out = out.transpose(1, -1)
        
        return out


class DenseGINConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, eps=0, train_eps=True):
        super().__init__()
        
        # TODO: Multi-layer model
        self.mlp = Group_Linear(in_channels, out_channels, groups, bias=False)
        
        self.init_eps = eps
        if train_eps:
            self.eps = Parameter(Tensor([eps]))
        else:
            self.register_buffer('eps', Tensor([eps]))
            
        self.reset_parameters()
            
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.init_eps)
        
    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[..., idx, idx] += 1
        
        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
        return adj
        
        
    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        B, C, N, _ = x.size()
        G = adj.size(0)
        
        # adj-norm
        adj = self.norm(adj, add_loop=False)
        
        # x: [B, C, G, N, F//G]
        x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        
        out = torch.matmul(adj, x)
        
        # DYNAMIC
        x_pre = x[:, :, :-1, ...]
        
        # out = x[:, :, 1:, ...] + x_pre
        out[:, :, 1:, ...] = out[:, :, 1:, ...] + x_pre
        # out = torch.cat( [x[:, :, 0, ...].unsqueeze(2), out], dim=2 )
        
        if add_loop:
            out = (1 + self.eps) * x + out
        
        # out: [B, C, G, N, F//G]
        out = self.mlp(out, True)
        
        # out: [B, C, N, F]
        C = out.size(1)
        out = out.transpose(2, 3).reshape(B, C, N, -1)
        
        return out


class Dense_TimeDiffPool2d(nn.Module):
    
    def __init__(self, pre_nodes, pooled_nodes, kern_size, padding):
        super().__init__()
        
        # TODO: add Normalization
        self.time_conv = nn.Conv2d(pre_nodes, pooled_nodes, (1, kern_size), padding=(0, padding))
        
        self.re_param = Parameter(Tensor(kern_size, 1))
        
    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')
        
        
    def forward(self, x: Tensor, adj: Tensor):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        x = x.transpose(1, 2)
        out = self.time_conv(x)
        out = out.transpose(1, 2)
        
        # s: [ N^(l+1), N^l, 1, K ]
        s = torch.matmul(self.time_conv.weight, self.re_param).view(out.size(-2), -1)

        # TODO: fully-connect, how to decrease time complexity
        out_adj = torch.matmul(torch.matmul(s, adj), s.transpose(0, 1))
        
        return out, out_adj

def get_att(x, W, emb_dim, batch_size):

    temp = torch.mean(x, 1).view((batch_size, 1, -1))  # (1, D)
    h_avg = torch.tanh(torch.matmul(temp, W))
    att = torch.bmm(x, h_avg.transpose(2, 1))
    output = torch.bmm(att.transpose(2, 1), x)

    return output

def glorot(shape, use_cuda):
    """Glorot & Bengio (AISTATS 2010) init."""
    if use_cuda == 1:
        rtn = nn.Parameter(torch.Tensor(*shape).cuda())
    else:
        rtn = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(rtn)
    return rtn

class Adaptive_Pooling_Layer(nn.Module):
    """ Memory_Pooling_Layer introduced in the paper 'MEMORY-BASED GRAPH NETWORKS' by Amir Hosein Khasahmadi, etc"""

    ### This layer is for downsampling a node set from N_input to N_output
    ### Input: [B,N_input,Dim_input]
    ###         B:batch size, N_input: number of input nodes, Dim_input: dimension of features of input nodes
    ### Output:[B,N_output,Dim_output]
    ###         B:batch size, N_output: number of output nodes, Dim_output: dimension of features of output nodes

    def __init__(self, Heads, Dim_input, N_output, Dim_output, use_cuda):#(4,128,1,128)
        """
            Heads: number of memory heads
            N_input : number of nodes in input node set
            Dim_input: number of feature dimension of input nodes
            N_output : number of the downsampled output nodes
            Dim_output: number of feature dimension of output nodes
        """
        super(Adaptive_Pooling_Layer, self).__init__()
        self.Heads = Heads#4
        self.Dim_input = Dim_input#128to256
        # self.left_num_nodes = []
        # for layer in range(3 + 1):
        #     left_node = round(24 * (1 - (0.2 * layer)))
        #     if left_node > 0:
        #         self.left_num_nodes.append(left_node)
        #     else:
        #         self.left_num_nodes.append(1)
        self.N_output = N_output#1
        self.Dim_output = Dim_output#128to256
        self.use_cuda = use_cuda

        if self.use_cuda == 1:
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        # Randomly initialize centroids
        self.centroids = nn.Parameter(2 * torch.rand(self.Heads, self.N_output, Dim_input) - 1)
        self.centroids.requires_grad = True

        hiden_channels = self.Heads
        if self.use_cuda == 1:
            self.input2centroids_weight = torch.nn.Parameter(
                torch.zeros(hiden_channels, 1).float().to(self.device), requires_grad=True)
            self.input2centroids_bias = torch.nn.Parameter(
                torch.zeros(hiden_channels).float().to(self.device), requires_grad=True)
        else:
            self.input2centroids_weight = torch.nn.Parameter(
                torch.zeros(hiden_channels, 1).float(), requires_grad=True)
            self.input2centroids_bias = torch.nn.Parameter(
                torch.zeros(hiden_channels).float(), requires_grad=True)

        self.input2centroids_ = nn.Sequential(nn.Linear(hiden_channels, self.Heads * self.N_output), nn.ReLU())

        self.memory_aggregation = nn.Conv2d(self.Heads, 1, [1, 1])

        self.dim_feat_transformation = nn.Linear(self.Dim_input, self.Dim_output)

        self.similarity_compute = torch.nn.CosineSimilarity(dim=4, eps=1e-6)

        self.relu = nn.ReLU()

        self.emb_dim = Dim_input#128
        self.W_0 = glorot([self.emb_dim, self.emb_dim], self.use_cuda)

    def forward(self, node_set, adj, zero_tensor=torch.tensor([0])):
        """
            node_set: Input node set in form of [batch_size, N_input, Dim_input]
            adj: adjacency matrix for node set x in form of [batch_size, N_input, N_input]
            zero_tensor: zero_tensor of size [1]

            (1): new_node_set = LRelu(C*node_set*W)
            (2): C = softmax(pixel_level_conv(C_heads))
            (3): C_heads = t-distribution(node_set, centroids)
            (4): W is a trainable linear transformation
        """

        node_set = node_set.transpose(1,2)#(180,24,128)
        node_set = torch.mean(node_set, dim=3)
        node_set_input = node_set#(180,24,128)
        batch_size, self.N_input, _ = node_set.size()#保存输入节点特征，并获取批大小 batch_size 和输入节点数量 N_input

        # batch_centroids = torch.mean(node_set,dim=1,keepdim=True)
        #使用 get_att 函数计算注意力机制，得到节点集的质心 batch_centroids
        batch_centroids = get_att(node_set, self.W_0, self.emb_dim, batch_size)#(180,1,128)
        #调整 batch_centroids 的维度顺序
        batch_centroids = batch_centroids.permute(0, 2, 1)#(180,128,1)
        #对质心应用线性变换和 ReLU 激活函数,得到（180，128，4）
        batch_centroids = torch.relu(
            torch.nn.functional.linear(batch_centroids, self.input2centroids_weight, self.input2centroids_bias))
        #通过一个序列化的网络模块（包含线性层和 ReLU 激活函数）进一步处理质心
        batch_centroids = self.input2centroids_(batch_centroids)
        #调整 batch_centroids 的维度，以匹配节点集的维度。，得到（180，4，1，128）
        batch_centroids = batch_centroids.permute(0, 2, 1).view(node_set.size()[0], self.Heads, self.N_output,
                                                                self.Dim_input)
        #将节点集扩展到包含多个“头”（heads）的维度
        # From initial node set [batch_size, N_input, Dim_input] to size [batch_size, Heads, N_output, N_input, Dim_input]
        node_set = torch.unsqueeze(node_set, 1).repeat(1, batch_centroids.shape[1], 1, 1)
        node_set = torch.unsqueeze(node_set, 2).repeat(1, 1, batch_centroids.shape[2], 1, 1)
        # Broadcast centroids to the same size as that of node set [batch_size, Heads, N_output, N_input, Dim_input]
        batch_centroids = torch.unsqueeze(batch_centroids, 3).repeat(1, 1, 1, node_set.shape[3], 1)

        # Compute the distance between original node set to centroids
        # [batch_size, Heads, N_output, N_input]
        C_heads = self.similarity_compute(node_set, batch_centroids)

        normalizer = torch.unsqueeze(torch.sum(C_heads, 2), 2)
        C_heads = C_heads / (normalizer + 1e-10)

        # Apply pixel-level convolution and softmax to C_heads
        # Get C: [batch_size, N_output, N_input]
        C = self.memory_aggregation(C_heads)
        # C = torch.softmax(C,1)

        C = C.squeeze(1)

        # [batch_size, N_output, N_input] * [batch_size, N_input, Dim_input] --> [batch_size, N_output, Dim_input]
        new_node_set = torch.matmul(C, node_set_input)
        # [batch_size, N_output, Dim_input] * [batch_size, Dim_input, Dim_output] --> [batch_size, N_output, Dim_output]
        new_node_set = self.dim_feat_transformation(new_node_set)

        """
            Calculate new_adj
        """

        # [batch_size, N_output, N_input] * [batch_size, N_input, N_input] --> [batch_size, N_output, N_input]
        # q_adj = torch.matmul(C, adj)

        # [batch_size, N_output, N_input] * [batch_size, N_input, N_output] --> [batch_size, N_output, N_output]
        # new_adj = self.relu(torch.matmul(q_adj, C.transpose(1, 2)))

        return new_node_set