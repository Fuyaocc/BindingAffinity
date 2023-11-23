import os
import re
import torch
import math
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader,Data
from utils.parse_args import get_args
from utils.MyDataset import MyGCNDataset,gcn_pickfold
from utils.run_epoch import mpnn_train,gcn_predict
from models.affinity_net_mpnn import Net

if __name__ == '__main__':
    args=get_args()
    print(args)

    ligand_complex_dict={}
    with open('./data/ligand.txt', 'r') as f:
        for line in f:
            pkd=line[19:23]
            ligand_name=line[0:4]
            ligand_complex_dict[ligand_name]=float(pkd)
    
    ligand_complex_list=set()

    ligands=os.listdir('./data/atom_graph/ligand/')

    for ligand in ligands:
        if ligand[:4] in ligand_complex_dict.keys():
            ligand_complex_list.add(ligand[:4])
    
    featureList=[]
    labelList=[]
    
    t=100
    for pdbname in ligand_complex_list:
        if t < 0:
            continue
        logging.info("load ligand data graph :"+pdbname)
        x = torch.load('./data/atom_graph/ligand/'+pdbname+"_x"+'.pth').to(torch.float)
        edge_index=torch.load('./data/atom_graph/ligand/'+pdbname+"edge_index"+'.pth').to(torch.int64)
        edge_attr=torch.load('./data/atom_graph/ligand/'+pdbname+"_edge_attr"+'.pth').to(torch.float)
        if(x.shape[0]==0 or edge_index.shape[0]==0 or edge_attr.shape[0]==0 ):
            continue
        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=ligand_complex_dict[pdbname],distillation_y=[0],name=pdbname,energy=[0])
        featureList.append(data)
        labelList.append(ligand_complex_dict[pdbname])
        t -= 1

        
    for i in range(5):
        net=Net(input_dim=args.dim
                        ,hidden_dim=64
                        ,output_dim=64)
        net.to(args.device)
        net.load_state_dict(torch.load(f"./models/saved/gcn/affinity_model{i}_dim{args.dim}_foldx.pt"))
        dataset=MyGCNDataset(featureList)
        dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
        criterion = torch.nn.MSELoss()
        
        names, test_prelist, test_truelist,test_loss = gcn_predict(net,dataloader,criterion,args.device,i,0)
        df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
        mae=F.l1_loss(torch.tensor(test_prelist),torch.tensor(test_truelist))
        mse=F.mse_loss(torch.tensor(test_prelist),torch.tensor(test_truelist))
        rmse=math.sqrt(mse)
        test_pcc = df.pre.corr(df.label)
        # print("Test Loss = %.4f"%(test_loss)+" , mse = %.4f"%(mse)+" , rmse = %.4f"%(rmse)+" , mae = %.4f"%(mae)+" , pcc = %.4f"%(test_pcc))
        with open(f'./tmp/pred/result_{i}.txt','w') as f:
            for j in range(0,len(test_truelist)):
                f.write(names[j])
                f.write('\t')
                f.write(str(test_truelist[j]))
                f.write('\t')
                f.write(str(test_prelist[j]))
                f.write('\n')
            
        logging.info(str(i)+" ,  MSE:"+str(mse)+" , rmse:"+str(rmse)+" , mae:"+str(mae)+" , PCC:"+str(test_pcc))
        