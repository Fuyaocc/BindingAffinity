import os
import re
import math
import torch
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import pandas as pd
import pickle
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from loss.pcc_loss import PCCLoss
from utils.parse_args import get_args
from utils.MyDataset import MyGCNDataset,gcn_pickfold
from utils.run_epoch import mpnn_train,gcn_predict
from utils.readFoldX import readFoldXResult
from models.affinity_net_mpnn import Net
from sklearn.preprocessing import StandardScaler
from datetime import datetime

if __name__ == '__main__':
    # torch.set_num_threads(2)

    args=get_args()
    print(args)

    complexdict={}
    val_feature = []
    for i in range(5):
        k_set = []
        for  line in open(args.inputdir+f'ddg_data/M4005_{i}.txt'):
            blocks=re.split('\t|\n|    ',line)
            pdbname = blocks[0]
            muts = blocks[1].split(',')
            for i in range(len(muts)):
                if i==0:
                    pdbname+='-'
                else:
                    pdbname+='+'
                pdbname+=muts[i]
            k_set.append(pdbname)
        val_feature.append(k_set)

    for line in open(args.inputdir+'dg_data/skempi_data.txt'):
        blocks=re.split('\t|\n|    ',line)
        pdbname=blocks[0]
        complexdict[pdbname]=float(blocks[1])
    
    
    exist_files = os.listdir(args.featdir)
    graph_set = set()
    for file in exist_files:
        graph_set.add(file.split('.')[0])
        
    featureList=[]
    
    for i in range(5):
        tmp = []
        for pdbname in val_feature[i]:
            if pdbname+'_x' not in graph_set: continue
            logging.info("load pdbbind data graph :"+pdbname)
            x = torch.load(args.featdir+pdbname+"_x"+'.pth').to(torch.float32)
            edge_index=torch.load(args.featdir+pdbname+"_edge_index"+'.pth').to(torch.int64)
            edge_attr=torch.load(args.featdir+pdbname+"_edge_attr"+'.pth').to(torch.float32)
            if os.path.exists(args.foldxdir+'energy/'+pdbname+"_energy"+'.pth') == False:
                try:
                    energy=readFoldXResult(args.foldxdir+'foldx_result/',pdbname.upper())
                except Exception:
                    energy=[0.]*22
                energy=torch.tensor(energy,dtype=torch.float32)
                torch.save(energy.to(torch.device('cpu')),args.foldxdir+'energy/'+pdbname+'_energy.pth')
            energy=torch.load(args.foldxdir+'energy/'+pdbname+"_energy"+'.pth').to(torch.float32)
            if x.shape[0] == 0 :continue        
            y = torch.tensor([complexdict[pdbname]])
            idx=torch.isnan(x)
            x[idx] = 0.0
            idx = torch.isinf(x)
            x[idx] = float(0.0)
            energy = energy[:21]
            idx = torch.isnan(energy)
            energy[idx] = 0.0
            pos = x[:, -6:-3]
            x = torch.cat([x[:,:10],x[:, 31:]],dim=1)
            x = torch.cat([x[:, :(-6)], x[:, (-3):]], dim=1)
            t = torch.abs(edge_attr)
            data = Data(x=x, edge_index=edge_index,edge_attr=t,y=y,pos=pos,name=pdbname,energy=energy)
            tmp.append(data)
        featureList.append(tmp)
        logging.info(len(featureList))
    
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    k = 5
    #交叉验证
    kf = KFold(n_splits=k,random_state=43, shuffle=True)
    best_pcc = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mae = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_epoch = [0,0,0,0,0,0,0,0,0,0]
    for i in range(5):
        #preprocessing Standard 标准化
        val_set = featureList[i]
        train_set = []
        for j in range(5):
            if j!= i:
                train_set += featureList[j]
        scaler = StandardScaler()
        train_x_tensor = torch.cat([data.x for data in train_set], dim=0)
        train_x_array = train_x_tensor.cpu().numpy()
        train_x_array_standardized = scaler.fit_transform(train_x_array)
        train_x_tensor_standardized = torch.tensor(train_x_array_standardized, dtype=torch.float32)
        start_idx = 0
        for data in train_set:
            num_samples = data.x.size(0)
            end_idx = start_idx + num_samples
            data.x = train_x_tensor_standardized[start_idx:end_idx]
            data = data.to(args.device)
            start_idx = end_idx

        val_x_tensor = torch.cat([data.x for data in val_set], dim=0)
        val_x_array = val_x_tensor.cpu().numpy()
        val_x_array_standardized = scaler.transform(val_x_array)
        val_x_tensor_standardized = torch.tensor(val_x_array_standardized, dtype=torch.float32)
        start_idx = 0
        for data in val_set:
            num_samples = data.x.size(0)
            end_idx = start_idx + num_samples
            data.x = val_x_tensor_standardized[start_idx:end_idx]
            data = data.to(args.device)
            start_idx = end_idx

        net=Net(input_dim=args.dim
                        ,hidden_dim=64
                        ,output_dim=64)
        
        net.to(args.device)

        train_dataset = MyGCNDataset(train_set)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = MyGCNDataset(val_set)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        
        criterion = torch.nn.HuberLoss(delta=args.alpha)
        optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-2)
        writer = SummaryWriter(args.logdir+TIMESTAMP+'val'+str(i))
        
        pre_out = {}
        for epoch in range(args.epoch):
            #train
            net, names, train_prelist, train_truelist, train_loss = mpnn_train(net, train_dataloader, optimizer, criterion, args, epoch)
            df = pd.DataFrame({'label':train_truelist, 'pre':train_prelist})
            train_pcc = df.pre.corr(df.label)
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/pcc', train_pcc, epoch)
            logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss))
            
            #val
            names,val_prelist, val_truelist,val_loss = gcn_predict(net, val_dataloader, criterion, args)
            df = pd.DataFrame({'label':val_truelist, 'pre':val_prelist})
            val_pcc = df.pre.corr(df.label)
            val_mae = F.l1_loss(torch.tensor(val_prelist),torch.tensor(val_truelist))
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/pcc', val_pcc, epoch)
            logging.info("Epoch "+ str(epoch)+ ": val Loss = %.4f"%(val_loss)+ ", val mae = %.4f"%(val_mae))

            
            if math.fabs(val_pcc) > best_pcc[i]:
                best_pcc[i]=math.fabs(val_pcc)
                best_epoch[i]=epoch
                best_mae[i] = val_mae
                torch.save(net.state_dict(),f'{args.modeldir}skempi_gnn{i}_dim{args.dim}_foldx.pt')
                with open(f'{args.outdir}pred/result_{i}.txt','w') as f:
                    for j in range(0,len(val_truelist)):
                        f.write(names[j])
                        f.write('\t')
                        f.write(str(val_truelist[j]))
                        f.write('\t')
                        f.write(str(val_prelist[j]))
                        f.write('\n')
    
    pcc=0.
    mae=0.
    for i in range(k):
        pcc=pcc+best_pcc[i]
        mae=mae+best_mae[i]
        logging.info('val_'+str(i)+' best_pcc = %.4f'%(best_pcc[i])+' , best_mae = %.4f'%(best_mae[i])+' , best_epoch : '+str(best_epoch[i]))
    print('pcc  :   '+str(pcc/k))
    print('mae  :   '+str(mae/k))