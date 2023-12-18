import torch
import torch.nn as nn
import torch.nn.functional as F

class AffinityNet(nn.Module):
    def __init__(self,dims):
        super().__init__()

        self.encode = ProteinCNN(dims)
        self.decode = MLPDecoder(dims,256)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        x = -F.relu(x)
        return x     
    
class ProteinCNN(nn.Module):
    def __init__(self, dims):
        super(ProteinCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=dims[0], out_channels=dims[1], kernel_size=1)
        self.bn1 = nn.BatchNorm1d(dims[1])
        self.conv1 = nn.Conv1d(in_channels=dims[1], out_channels=dims[2], kernel_size=3)
        self.bn1 = nn.BatchNorm1d(dims[2])

    def forward(self, v):
        v = v.transpose(2, 1)
        v = F.leaky_relu(self.conv1(v))
        v = F.max_pool1d(v)
        return v
    
class MLPDecoder(nn.Module):
    def __init__(self, dims):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(dims[2], dims[3])
        self.bn1 = nn.BatchNorm1d(dims[3])
        self.fc2 = nn.Linear(dims[3], dims[4])
        self.bn2 = nn.BatchNorm1d(dims[4])
        self.fc3 = nn.Linear(dims[4], 1)

    def forward(self, x):
        x = self.bn1(F.leaky_relu(self.fc1(x)))
        x = F.dropout(x,0.1)
        x = self.bn2(F.leaky_relu(self.fc2(x)))
        x= F.dropout(x,0.1)
        x=self.fc3(x)
        return x
