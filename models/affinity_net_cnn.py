import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,dims):
        super().__init__()

        self.encode = ProteinCNN(dims)
        self.decode = MLPDecoder(dims)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x     
    
class ProteinCNN(nn.Module):
    def __init__(self, dims):
        super(ProteinCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=dims[0], out_channels=dims[1], kernel_size=1, stride=1)

    def forward(self, v):
        v = v.transpose(2, 1)
        v = F.leaky_relu(self.conv1(v))
        v = F.dropout(v,0.2)
        v = F.avg_pool1d(v,200)
        v = v.reshape(v.size(0), -1)
        return v
    
class MLPDecoder(nn.Module):
    def __init__(self, dims):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(dims[3], dims[4])
        self.bn1 = nn.BatchNorm1d(dims[4])
        self.fc2 = nn.Linear(dims[4], dims[5])
        self.bn2 = nn.BatchNorm1d(dims[5])
        self.fc3 = nn.Linear(dims[5], 1)

    def forward(self, x):
        x = self.bn1(F.leaky_relu(self.fc1(x)))
        x = F.dropout(x,0.2)
        x = self.bn2(F.leaky_relu(self.fc2(x)))
        x= F.dropout(x,0.2)
        x=self.fc3(x)
        return x
