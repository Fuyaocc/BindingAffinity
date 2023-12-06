import os
import re
import torch
import math
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils.parse_args import get_args
from utils.MyDataset import MyGCNDataset
from utils.run_epoch import gcn_predict
from utils.readFoldX import readFoldXResult
from models.affinity_net_mpnn import Net
from sklearn.preprocessing import MinMaxScaler, StandardScaler

if __name__ == '__main__':
    args=get_args()
    print(args)

    complex_dict={}
    with open(args.inputdir, 'r') as f:
        for line in f:
            blocks = re.split('\t|\n',line)
            ligand_name=blocks[0]
            complex_dict[ligand_name]=float(blocks[1])

    distillation_data = {}
    with open('./data/distillation_data.txt') as f:
        for line in f:
            v = re.split("\t|\n",line)
            distillation_data[v[0]]=float(v[1])
    
    complex_list=set()

    pdbs=os.listdir(args.featdir)

    for pdb in pdbs:
        if pdb[:4] in complex_dict.keys():
            complex_list.add(pdb[:4])
    
    featureList=[]
    labelList=[]
    scaler_zscore = StandardScaler()

    for pdbname in complex_list:

        logging.info("load ligand data graph :"+pdbname)
        x = torch.load(args.featdir+pdbname+"_x"+'.pth').to(torch.float)
        edge_index=torch.load(args.featdir+pdbname+"_edge_index"+'.pth').to(torch.int64)
        edge_attr=torch.load(args.featdir+pdbname+"_edge_attr"+'.pth').to(torch.float)
        if os.path.exists(args.featdir+pdbname+"_energy"+'.pth') == False:
            energy=readFoldXResult(args.foldxPath,pdbname)
            energy=torch.tensor(energy,dtype=torch.float32)
            torch.save(energy.to(torch.device('cpu')),args.featdir+pdbname+"_energy"+'.pth')
        energy=torch.load(args.featdir+pdbname+"_energy"+'.pth').to(torch.float).to(args.device)
        # print(x.shape)
        idx=torch.isnan(x)
        x[idx]=0.0
        idx = torch.isinf(x)
        x[idx] = float(0.0)
        x = scaler_zscore.fit_transform(x.numpy())
        x = torch.tensor(x).to(torch.float).to(args.device)
        energy = energy[:21]
        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=complex_dict[pdbname],distillation_y=distillation_data[pdbname],name=pdbname,energy=energy)
        featureList.append(data)
        labelList.append(complex_dict[pdbname])

    print(len(featureList))

    best_pcc = 0.0
    best_mse = 0.0
    for i in range(5):
        net=Net(input_dim=args.dim
                        ,hidden_dim=64
                        ,output_dim=32)
        net.to(args.device)
        net.load_state_dict(torch.load(f"./models/saved/gcn/affinity_model{i}_dim{args.dim}_foldx.pt"))
        dataset=MyGCNDataset(featureList)
        dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
        criterion = torch.nn.MSELoss()
        
        names, test_prelist, test_truelist,test_loss = gcn_predict(net,dataloader,criterion,args,i,0)
        df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
        mae=F.l1_loss(torch.tensor(test_prelist),torch.tensor(test_truelist))
        mse=F.mse_loss(torch.tensor(test_prelist),torch.tensor(test_truelist))
        rmse=math.sqrt(mse)
        test_pcc = df.pre.corr(df.label)
        with open(f'./tmp/pred/result_{i}.txt','w') as f:
            for j in range(0,len(test_truelist)):
                f.write(names[j])
                f.write('\t')
                f.write(str(test_truelist[j]))
                f.write('\t')
                f.write(str(test_prelist[j]))
                f.write('\n')
            
        logging.info(str(i)+" ,  MSE:"+str(mse)+" , rmse:"+str(rmse)+" , mae:"+str(mae)+" , PCC:"+str(test_pcc))
        best_pcc += test_pcc
        best_mse += test_loss
    print('pcc  :   '+str(best_pcc/5))
    print('mse  :   '+str(best_mse/5))
        