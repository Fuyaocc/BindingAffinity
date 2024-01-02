import os
import re
import torch
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import pandas as pd
import pickle
import math
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.parse_args import get_args
from utils.MyDataset import MyDataset,pickfold
from utils.run_epoch import cnn_train,cnn_predict
from utils.resFeature import getAAOneHotPhys
from utils.readFoldX import readFoldXResult
from models.affinity_net_cnn import Net
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from datetime import datetime

if __name__ == '__main__':
    args=get_args()
    print(args)

    complexdict={} # pdbname : [seq1, seq2, bingding_affinity]
    
    for line in open(args.inputdir+'pdbbind_data.txt'):
        blocks=re.split('\t|\n|    ',line)
        pdbname=blocks[0]
        complexdict[pdbname]=float(blocks[1])
    
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
    namelist=[]
    maxlen = 0
    for pdbname in complexdict.keys():
        if pdbname in filter_set :continue 
        #local redisue
        if pdbname in graph_dict:
            logging.info("load pdbbind data graph :"+pdbname)
            # if os.path.exists(args.energydir+pdbname+"_energy"+'.pth') == False:
            #     energy=readFoldXResult(args.foldxPath,pdbname)
            #     energy=torch.tensor(energy,dtype=torch.float32)
            #     torch.save(energy.to(torch.device('cpu')),args.featdir+pdbname+"_energy"+'.pth')
            # energy=torch.load(args.energydir+pdbname+"_energy"+'.pth').to(torch.float32).to(args.device)
            # energy = energy[:21]
            x = torch.load(args.featdir+pdbname+"_x"+'.pth').to(torch.float32)
            idx=torch.isnan(x)
            x[idx]=0.0
            idx = torch.isinf(x)
            x[idx] = float(0.0)
            edge_index=torch.load(args.featdir+pdbname+"_edge_index"+'.pth').to(torch.int32)
            edge_attr=torch.load(args.featdir+pdbname+"_edge_attr"+'.pth').to(torch.float32)
            edge_dict = {}
            
            for k in range(0,len(edge_index[0])):
                if k%2 == 1:
                    continue
                i = edge_index[0][k]
                j = edge_index[1][k]
                if edge_attr[k] > 0:
                    t = [1,0]+[math.fabs(edge_attr[k])]+x[i][:-1].tolist()+x[j][:-1].tolist()
                else:
                    t = [0,1]+[math.fabs(edge_attr[k])]+x[i][:-1].tolist()+x[j][:-1].tolist()
                edge_dict[math.fabs(float(edge_attr[k]))] = t
            
            sorted_keys = sorted(edge_dict.keys())
            x_len = 2*(x.shape[1]-1)+3
            xx = []
            if len(sorted_keys) < 200:
                padding_len = 200-len(sorted_keys)
                zero_matrix = [[0 for _ in range(x_len)] for _ in range(padding_len)]
                for i in range(0, len(sorted_keys)):
                    xx.append(edge_dict[sorted_keys[i]])
                xx += zero_matrix
            else:
                for i in range(0,200):
                    xx.append(edge_dict[sorted_keys[i]])
            xx = torch.tensor(xx,dtype=torch.float32).to(args.device)
            featureList.append(xx)
            labelList.append(complexdict[pdbname])
            namelist.append(pdbname)
    logging.info(len(featureList))
    # #交叉验证
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    kf = KFold(n_splits=5,random_state=43, shuffle=True)
    best_pcc = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mae = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_epoch = [0,0,0,0,0,0,0,0,0,0]
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        x_train,y_train,name_train,x_test,y_test,name_test = pickfold(featureList,labelList,namelist, train_index, test_index)

        dims = [263,131,64,200,128,64]
        # dims = [131,131,131,50,128,64]
        net=Net(dims)
        
        net.to(args.device)

        train_dataset=MyDataset(x_train,y_train,name_train)
        train_dataloader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        val_dataset=MyDataset(x_test,y_test,name_test)
        val_dataloader=DataLoader(val_dataset, batch_size=args.batch_size//4, shuffle=True)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4, weight_decay = 1e-1)

        writer = SummaryWriter(args.logdir+TIMESTAMP+'val'+str(i))
        
        for epoch in range(args.epoch):
            #train
            net, train_prelist, train_truelist, train_loss= cnn_train(net, train_dataloader, optimizer, criterion,args)

            df = pd.DataFrame({'label':train_truelist, 'pre':train_prelist})
            train_pcc = df.pre.corr(df.label)
            train_mae = F.l1_loss(torch.tensor(train_prelist),torch.tensor(train_truelist))
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/pcc', train_pcc, epoch)
            logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss)+ ", train mae = %.4f"%(train_mae))
            
            #val
            val_name, val_prelist, val_truelist, val_loss = cnn_predict(net, val_dataloader, criterion, args, i, epoch)
            df = pd.DataFrame({'label':val_truelist, 'pre':val_prelist})
            val_pcc = df.pre.corr(df.label)
            val_mae = F.l1_loss(torch.tensor(val_prelist),torch.tensor(val_truelist))
            writer.add_scalar('val/val_loss', val_loss, epoch)
            writer.add_scalar('val/val_pcc', val_pcc, epoch)
            logging.info("Epoch "+ str(epoch)+ ": val Loss = %.4f"%(val_loss)+ ", val mae = %.4f"%(val_mae))
            if val_pcc > best_pcc[i]:
                best_pcc[i]=val_pcc
                best_epoch[i]=epoch
                best_mae[i] = val_mae
                torch.save(net.state_dict(),f'{args.modeldir}PPA_Pred_cnn{i}_dim{args.dim}_foldx.pt')
                with open(f'./tmp/pred/result_{i}.txt','w') as f:
                    for j in range(0,len(val_truelist)):
                        f.write(val_name[j])
                        f.write('\t')
                        f.write(str(val_truelist[j]))
                        f.write('\t')
                        f.write(str(val_prelist[j]))
                        f.write('\n')
    
    pcc=0.
    mae=0.
    for i in range(5):
        pcc=pcc+best_pcc[i]
        mae=mae+best_mae[i]
        logging.info('val_'+str(i)+' best_pcc = %.4f'%(best_pcc[i])+' , best_mae = %.4f'%(best_mae[i])+' , best_epoch : '+str(best_epoch[i]))
    print('pcc  :   '+str(pcc/5))
    print('mae  :   '+str(mae/5))
            
