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
    
    origin2clean = {}
    # for line in open(args.inputdir+'skempi_origin2clean.txt'):
    #     blocks=re.split('\t|\n',line)
    #     origin2clean[blocks[0]]=blocks[1]
    for line in open(args.inputdir+'dg_data/All_set.txt'):
        blocks=re.split('\t|\n|    ',line)
        pdbname=blocks[0]
        # if pdbname[-2:] == 'wt':
        #     pdbname = pdbname[:4].lower()
        # else:
        #     if pdbname in origin2clean.keys():
        #         pdbname = origin2clean[pdbname]
        complexdict[pdbname]=float(blocks[1])
    
    test_set=set()
    for line in open(args.inputdir+'dg_data/PPIAffinity_test.txt'):
        blocks=re.split('\t|\n|    ',line)
        test_set.add(blocks[0])
    
    
    exist_files = os.listdir(args.featdir)
    graph_set = set()
    for file in exist_files:
        graph_set.add(file[:4])
        
    featureList=[]
    labelList=[]
    testfeatureList = []
        
    for pdbname in complexdict.keys():
        if pdbname not in graph_set: continue
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
        #t = torch.zeros((len(edge_attr), 3))
        #t[edge_attr < 0, 1] = 1
        #t[edge_attr >= 0, 0] = 1
        # t[:,2] = torch.abs(edge_attr)
        t = torch.abs(edge_attr)
        #消融实验
        
        # x = x[:,:36] #AAindex
        # x = x[:,:43] #AAindex+dssp 
        # x = x[:,:75] #AAindex+dssp+esm1f1
        # x = x[:,:107] #AAindex+dssp+esm1f1+esm1v
        # energy.fill_(0.0) #去掉foldx
        # t.fill_(1.0) #去掉边信息
        # x[:,0:10] = 0.0 #去掉二分图
        
        data = Data(x=x, edge_index=edge_index,edge_attr=t,y=y,pos=pos,name=pdbname,energy=energy)
        if pdbname in test_set:
            testfeatureList.append(data)
        else:    
            featureList.append(data)
            labelList.append(complexdict[pdbname])
    logging.info(len(featureList))
    
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    k = 5
    #交叉验证
    kf = KFold(n_splits=k,random_state=43, shuffle=True)
    best_pcc = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_spcc = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mae = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_test_pcc = [0.0,0.0,0.0,0.0,0.0]
    best_test_spcc = [0.0,0.0,0.0,0.0,0.0]
    best_test_mae = [0.0,0.0,0.0,0.0,0.0]
    best_epoch = [0,0,0,0,0,0,0,0,0,0]
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        #preprocessing Standard 标准化
        train_set,val_set=gcn_pickfold(featureList, train_index, test_index)
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

        # with open(f'./tmp/scaler/standard_scaler{i}.picke','wb') as sc:
        #     pickle.dump(scaler, sc)

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
        
        test_x_tensor = torch.cat([data.x for data in testfeatureList], dim=0)
        test_x_array = test_x_tensor.cpu().numpy()
        test_x_array_standardized = scaler.transform(test_x_array)
        test_x_tensor_standardized = torch.tensor(test_x_array_standardized, dtype=torch.float32)
        start_idx = 0
        for data in testfeatureList:
            num_samples = data.x.size(0)
            end_idx = start_idx + num_samples
            data.x = test_x_tensor_standardized[start_idx:end_idx]
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
        test_dataset = MyGCNDataset(testfeatureList)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        
        criterion = torch.nn.HuberLoss(delta=args.alpha)
        optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-2)
        writer = SummaryWriter(args.logdir+TIMESTAMP+'val'+str(i))
        
        pre_out = {}
        for epoch in range(args.epoch):
            #train
            net, names, train_prelist, train_truelist, train_loss = mpnn_train(net, train_dataloader, optimizer, criterion, args, epoch)
            df = pd.DataFrame({'label':train_truelist, 'pre':train_prelist})
            train_pcc = df.pre.corr(df.label,method='pearson')
            train_spcc = df.pre.corr(df.label,method='spearman')
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/pcc', train_pcc, epoch)
            writer.add_scalar('train/spcc', train_spcc, epoch)
            logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss))
            
            #val
            names,val_prelist, val_truelist,val_loss = gcn_predict(net, val_dataloader, criterion, args)
            df = pd.DataFrame({'label':val_truelist, 'pre':val_prelist})
            val_pcc = df.pre.corr(df.label,method='pearson')
            val_spcc = df.pre.corr(df.label,method='spearman')
            val_mae = F.l1_loss(torch.tensor(val_prelist),torch.tensor(val_truelist))
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/pcc', val_pcc, epoch)
            writer.add_scalar('val/spcc', val_spcc, epoch)
            logging.info("Epoch "+ str(epoch)+ ": val Loss = %.4f"%(val_loss)+ ", val mae = %.4f"%(val_mae))
            
            #test
            _,test_prelist, test_truelist,_ = gcn_predict(net, test_dataloader, criterion, args)
            df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
            test_pcc = df.pre.corr(df.label,method='pearson')
            test_spcc = df.pre.corr(df.label,method='spearman')
            test_mae = F.l1_loss(torch.tensor(test_prelist),torch.tensor(test_truelist))
            
            if math.fabs(val_pcc) > best_pcc[i]:
                best_pcc[i]=math.fabs(val_pcc)
                best_epoch[i]=epoch
                best_mae[i] = val_mae
                best_test_spcc[i] = math.fabs(test_spcc)
                best_test_pcc[i] = test_pcc
                best_test_mae[i] = test_mae
                torch.save(net.state_dict(),f'{args.modeldir}PPA_Pred_gnn{i}_dim{args.dim}_foldx.pt')
                with open(f'{args.outdir}pred/result_{i}.txt','w') as f:
                    for j in range(0,len(val_truelist)):
                        f.write(names[j])
                        f.write('\t')
                        f.write(str(val_truelist[j]))
                        f.write('\t')
                        f.write(str(val_prelist[j]))
                        f.write('\n')
    
    pcc = 0.
    spcc = 0.
    mae = 0.
    pcc1 = 0.
    spcc1 = 0.
    mae1 = 0.
    for i in range(k):
        pcc=pcc+best_pcc[i]
        spcc=spcc+best_spcc[i]
        mae=mae+best_mae[i]
        pcc1=pcc1+best_test_pcc[i]
        spcc1 = spcc1+best_test_spcc[i]
        mae1=mae1+best_test_mae[i]
        logging.info('val_'+str(i)+' best_pcc = %.4f'%(best_pcc[i])+' best_spcc = %.4f'%(best_spcc[i])+' , best_mae = %.4f'%(best_mae[i])+' , best_epoch : '+str(best_epoch[i]))
    print('pcc  :   '+str(pcc/k))
    print('spcc  :   '+str(spcc/k))
    print('mae  :   '+str(mae/k))
    print('pcc1  :   '+str(pcc1/5))
    print('spcc1  :   '+str(spcc1/5))
    print('mae1  :   '+str(mae1/5))