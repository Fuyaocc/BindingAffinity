import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,global_mean_pool,global_max_pool
from torch_geometric.utils import add_self_loops, degree

class MPNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim,drop_weight):
        super(MPNN, self).__init__(aggr='add', flow='source_to_target')
        self.lin = torch.nn.Sequential(
            nn.Linear(input_dim , hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(drop_weight),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # 添加自环边
        edge_index, edge_attr = add_self_loops(edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))

        x = self.lin(x)

        # 计算归一化系数
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 节点特征传递
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j,edge_attr, norm):
        return norm.view(-1, 1) * edge_attr.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
    
    def aggregate(self, inputs, index, ptr, dim_size):
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        
        self.mpnn_0 = MPNN(input_dim=input_dim,hidden_dim=hidden_dim, output_dim=hidden_dim, drop_weight=0.1)
        self.mpnn_1 = MPNN(input_dim=hidden_dim,hidden_dim=hidden_dim, output_dim=output_dim, drop_weight=0.2)
        
        self.emb = torch.nn.Sequential(
            nn.Linear(21, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU()
        )

        self.lin = torch.nn.Sequential(
            nn.Linear(output_dim+8, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(output_dim, 1),
        )

    def forward(self, data, mode=True):
        x = self.mpnn_0(data.x, data.edge_index, data.edge_attr)
        x = F.leaky_relu(x)
        x = self.mpnn_1(x, data.edge_index, data.edge_attr)
        x = F.leaky_relu(x)
        x = global_max_pool(x, data.batch)
        e = data.energy.reshape(x.shape[0],21)
        e = self.emb(e)
        x = torch.cat([x,e],dim=1)
        x = self.lin(x)
        x = -F.relu(x)
        return x