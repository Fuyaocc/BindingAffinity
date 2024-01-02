import shap
import torch
import pickle

from utils.parse_args import get_args
from models.affinity_net_mpnn import Net
from torch_geometric.data import Data,Batch


if __name__ == '__main__':
    args = get_args()
    pdbname = ' '
    ppa = 1.0
    i = 0
    x = torch.load(args.featdir+pdbname+"_x"+'.pth').to(torch.float32)
    edge_index=torch.load(args.featdir+pdbname+"_edge_index"+'.pth').to(torch.int32).to(args.device)
    edge_attr=torch.load(args.featdir+pdbname+"_edge_attr"+'.pth').to(torch.float32).to(args.device)
    energy=torch.load(args.foldxdir+'energy/'+pdbname+"_energy"+'.pth').to(torch.float32).to(args.device)
    idx=torch.isnan(x)
    x[idx]=0.0
    idx = torch.isinf(x)
    x[idx] = float(0.0)
    x.to(args.device)
    energy = energy[:21]
    idx = torch.isnan(energy)
    energy[idx] = 0.0
    data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=ppa,y_soft=ppa,name=pdbname,energy=energy)
    data.to(args.device)
    net=Net(input_dim=args.dim
                        ,hidden_dim=64
                        ,output_dim=32)
    net.to(args.device)
    net.load_state_dict(torch.load(args.modeldir+f"affinity_model{i}_dim{args.dim}_foldx.pt"))


    explainer = shap.DeepExplainer(net,data)

    shap_values = explainer.shap_values(data)

    print(shap_values)