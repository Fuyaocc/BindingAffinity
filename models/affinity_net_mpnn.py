import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing,global_mean_pool
from torch_geometric.utils import add_self_loops, degree

class MPNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim,drop_weight):
        super(MPNN, self).__init__(aggr='add')
        self.lin = torch.nn.Sequential(
            nn.Linear(input_dim , hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(drop_weight),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        # 添加自环边
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))

        # 计算归一化系数
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 节点特征传递
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * edge_attr.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        
        self.mpnn = MPNN(input_dim=input_dim,hidden_dim=hidden_dim, output_dim=output_dim, drop_weight=0.1)
        self.lin = torch.nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(output_dim, 1),
        )

    def forward(self, data, mode=True, device='cuda:0'):
        x = self.mpnn(data)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        # e = data.energy.reshape(x.shape[0],21)
        # x = torch.cat([x,e],dim=1)
        x = self.lin(x)
        
        return x