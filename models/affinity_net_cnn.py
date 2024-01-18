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
        self.conv1 = nn.Conv1d(in_channels=dims[0], out_channels=dims[0], kernel_size=1, stride=1)

    def forward(self, v):
        v = v.transpose(2, 1)
        v = F.leaky_relu(self.conv1(v))
        v = v.transpose(2, 1)
        v = F.max_pool1d(v,2)
        return v
    
class MLPDecoder(nn.Module):
    def __init__(self, dims):
        super(MLPDecoder, self).__init__()
        self.tolow = nn.Sequential(
            nn.Linear(dims[1],dims[2]),
            nn.LeakyReLU(),
            nn.Linear(dims[2],1)
        )
        
        self.fc1 = nn.Linear(dims[3], dims[4])
        self.bn1 = nn.BatchNorm1d(dims[4])
        self.fc2 = nn.Linear(dims[4], dims[5])
        self.bn2 = nn.BatchNorm1d(dims[5])
        self.fc3 = nn.Linear(dims[5], 1)

    def forward(self, x):
        x = self.tolow(x)
        x = x.reshape(x.size(0), -1)
        x = self.bn1(F.leaky_relu(self.fc1(x)))
        x = self.bn2(F.leaky_relu(self.fc2(x)))
        x = F.dropout(x,0.3)
        x=self.fc3(x)
        return x
