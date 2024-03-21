import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm

class Net(nn.Module):
    def __init__(self,dim=256):
        super().__init__()

        self.encode=ProteinCNN([dim,dim//2,dim//4])
        self.encode1=ProteinCNN([dim,dim//2,dim//4])
        
        self.bcn = weight_norm(BANLayer(v_dim=dim//4,q_dim=dim//4, h_dim=dim//4, h_out=2) ,
                               name='h_mat', 
                               dim=None)

        self.decode = MLPDecoder(dim//4,32)
        
    def forward(self, c1, c2):
        c1 = self.encode(c1)
        c2 = self.encode1(c2)
        f, att = self.bcn(c1, c2)
         
        score = self.decode(f)
        return score
        
    
class ProteinCNN(nn.Module):
    def __init__(self, in_ch):
        super(ProteinCNN, self).__init__()
        self.conv1ds=nn.Sequential(
                nn.Conv1d(in_channels=in_ch[0],out_channels=in_ch[1],kernel_size=3,stride=1),#(batch_size,dim,len-2)
                nn.LeakyReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(in_channels=in_ch[1],out_channels=in_ch[2],kernel_size=6,stride=1),#(batch_size,dim,len-2)
                nn.LeakyReLU(),
                nn.MaxPool1d(2)
                )

        self.gru1=nn.GRU(247, 32, batch_first=True, bidirectional= True)


    def forward(self, v):
        v = v.permute(0, 2, 1)
        v = self.conv1ds(v)
        v,_ = self.gru1(v)
        v = F.leaky_relu(v)
        v = v.permute(0, 2, 1)
        # print(v.shape)
        # v = v.view(v.size(0), v.size(2), -1)
        return v
    
class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x
