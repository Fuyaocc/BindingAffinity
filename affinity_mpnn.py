import os
import re
import torch
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import pandas as pd
import pickle
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
from utils.parse_args import get_args
from utils.MyDataset import MyGCNDataset,gcn_pickfold
from utils.run_epoch import mpnn_train,gcn_predict
from utils.resFeature import getAAOneHotPhys
from utils.readFoldX import readFoldXResult
from models.affinity_net_mpnn import Net
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def collate_fn(data_list):
    return Batch.from_data_list(data_list)

if __name__ == '__main__':
    args=get_args()
    print(args)

    complexdict={} # pdbname : [seq1, seq2, bingding_affinity]

    for line in open(args.inputdir+'pdbbind_data.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        complexdict[pdbname]=float(blocks[1])
    
    test_set=set()
    for line in open(args.inputdir+'test_set.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        test_set.add(pdbname)
    
    distillation_data = {}
    with open(args.inputdir+'distillation_data.txt') as f:
        for line in f:
            v = re.split("\t|\n",line)
            distillation_data[v[0]]=float(v[1])
    
    filter_set = set()#can't cal dssp or seq too long
    with open(args.inputdir+'filter_set.txt') as f:
        for line in f:
            filter_set.add(line[:-1])

    files=os.listdir(args.featdir)
    graph_dict=set()
    for file in files:
        graph_dict.add(file.split("_")[0])
        
    resfeat=getAAOneHotPhys()

    featureList=[]
    labelList=[]
    test_featureList=[]
    test_labelList=[]
    
    for pdbname in complexdict.keys():
        if pdbname in filter_set or pdbname in test_set:continue 
        #local redisue
        if pdbname in graph_dict:
            logging.info("load pdbbind data graph :"+pdbname)
            x = torch.load(args.featdir+pdbname+"_x"+'.pth').to(torch.float32)
            edge_index=torch.load(args.featdir+pdbname+"_edge_index"+'.pth').to(torch.int32).to(args.device)
            edge_attr=torch.load(args.featdir+pdbname+"_edge_attr"+'.pth').to(torch.float32).to(args.device)
            if os.path.exists(args.featdir+pdbname+"_energy"+'.pth') == False:
                energy=readFoldXResult(args.foldxPath,pdbname)
                energy=torch.tensor(energy,dtype=torch.float32)
                torch.save(energy.to(torch.device('cpu')),args.featdir+pdbname+"_energy"+'.pth')
            energy=torch.load(args.featdir+pdbname+"_energy"+'.pth').to(torch.float32).to(args.device)
            idx=torch.isnan(x)
            x[idx]=0.0
            idx = torch.isinf(x)
            x[idx] = float(0.0)
            x.to(args.device)
            energy = energy[:21]
            data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=complexdict[pdbname],distillation_y=distillation_data[pdbname],name=pdbname,energy=energy)

            featureList.append(data)
            labelList.append(complexdict[pdbname])

    logging.info(len(featureList))
    #交叉验证
    kf = KFold(n_splits=5,random_state=2023, shuffle=True)
    best_pcc = [0.0,0.0,0.0,0.0,0.0]
    best_mse = [0.0,0.0,0.0,0.0,0.0]
    best_epoch = [0,0,0,0,0]
    scaler_params = []
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        #preprocessing Standard 标准化
        train_set,val_set=gcn_pickfold(featureList, train_index, test_index)

        scaler = StandardScaler()
        train_x_tensor = torch.cat([data.x for data in train_set], dim=0)
        train_x_array = train_x_tensor.numpy()
        train_x_array_standardized = scaler.fit_transform(train_x_array)
        train_x_tensor_standardized = torch.tensor(train_x_array_standardized, dtype=torch.float32)
        start_idx = 0
        for data in train_set:
            num_samples = data.x.size(0)
            end_idx = start_idx + num_samples
            data.x = train_x_tensor_standardized[start_idx:end_idx]
            start_idx = end_idx
        
        with open(f'./tmp/{args.preprocess+str(i)}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        val_x_tensor = torch.cat([data.x for data in val_set], dim=0)
        val_x_array = val_x_tensor.numpy()
        val_x_array_standardized = scaler.transform(val_x_array)
        val_x_tensor_standardized = torch.tensor(val_x_array_standardized, dtype=torch.float32)
        start_idx = 0
        for data in val_set:
            num_samples = data.x.size(0)
            end_idx = start_idx + num_samples
            data.x = val_x_tensor_standardized[start_idx:end_idx]
            start_idx = end_idx

        net=Net(input_dim=args.dim
                        ,hidden_dim=64
                        ,output_dim=32)
        
        net.to(args.device)

        train_dataset=MyGCNDataset(train_set)
        train_dataloader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        val_dataset=MyGCNDataset(val_set)
        val_dataloader=DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        criterion = torch.nn.MSELoss()
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-3)

        writer = SummaryWriter(args.logdir+str(i))
        
        for epoch in range(args.epoch):
            #train
            net, train_prelist, train_truelist, train_loss= mpnn_train(net, train_dataloader, optimizer, criterion,args, kl_loss)

            logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss))

            df = pd.DataFrame({'label':train_truelist, 'pre':train_prelist})
            train_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_train/loss', train_loss, epoch)
            writer.add_scalar('affinity_train/pcc', train_pcc, epoch)
            
            #val
            _,val_prelist, val_truelist,val_loss = gcn_predict(net, val_dataloader, criterion, args, i, epoch)
            df = pd.DataFrame({'label':val_truelist, 'pre':val_prelist})
            val_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_val/val_loss', val_loss, epoch)
            writer.add_scalar('affinity_val/val_pcc', val_pcc, epoch)
            logging.info("Epoch "+ str(epoch)+ ": val Loss = %.4f"%(val_loss))
            if val_pcc > best_pcc[i]:
                best_pcc[i]=val_pcc
                best_epoch[i]=epoch
                torch.save(net.state_dict(),f'./models/saved/gcn/affinity_model{i}_dim{args.dim}_foldx.pt')
    
    pcc=0.
    mse=0.
    for i in range(5):
        pcc=pcc+best_pcc[i]
        mse=mse+best_mse[i]
        logging.info('val_'+str(i)+' best_pcc = %.4f'%(best_pcc[i])+' , best_mse = %.4f'%(best_mse[i])+' , best_epoch : '+str(best_epoch[i]))

    np.save('./tmp/standard_params.npy',np.array(scaler_params))
    print('pcc  :   '+str(pcc/5))
    print('mse  :   '+str(mse/5))
            