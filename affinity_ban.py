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
from utils.run_epoch import cnn_train,cnn_predict
from utils.readFoldX import readFoldXResult
from models.affintiy_net_ban import Net
from sklearn.preprocessing import StandardScaler
from datetime import datetime

if __name__ == '__main__':
    # torch.set_num_threads(2)

    args=get_args()
    print(args)

    complexdict={}
    for line in open(args.inputdir+'dg_data/PPIAffinity_set.txt'):
        blocks=re.split('\t|\n|    ',line)
        pdbname=blocks[0]
        complexdict[pdbname]=float(blocks[1])

    pdb_mol = {}
    for line in open(args.inputdir+'dg_data/pdb_mols.txt'):
        line = line.strip('\t')
        blocks=re.split('\t|\n|    ',line)
        pdb_mol[blocks[0]] = blocks[1:]
    
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

    testlist = []

    for pdbname in complexdict.keys():
        x1=torch.load('../feats/esm/pdbbind/esm1v/'+pdbname.lower()+'_'+pdb_mol[pdbname][0]+'.pth')
        x2=torch.load('../feats/esm/pdbbind/esm1v/'+pdbname.lower()+'_'+pdb_mol[pdbname][0]+'.pth')
        x1=F.avg_pool1d(x1,5,5)
        x2=F.avg_pool1d(x2,5,5)
        x1 = torch.cat([x1,torch.zeros(1000-x1.shape[0],256)],dim=0).to(args.device)
        x2 = torch.cat([x2,torch.zeros(1000-x2.shape[0],256)],dim=0).to(args.device)
        if pdbname in  test_set:
            testlist.append([x1,x2,complexdict[pdbname], pdbname])
        else:
            featureList.append([x1,x2,complexdict[pdbname], pdbname])
            labelList.append(complexdict[pdbname])

    logging.info(len(featureList))
    
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    #交叉验证
    kf = KFold(n_splits=5,random_state=43, shuffle=True)
    best_pcc = [0.0,0.0,0.0,0.0,0.0]
    best_mae = [0.0,0.0,0.0,0.0,0.0]
    best_epoch = [0,0,0,0,0,0,0,0,0,0]
    best_test_pcc = [0.0,0.0,0.0,0.0,0.0]
    best_test_mae = [0.0,0.0,0.0,0.0,0.0]
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        #preprocessing Standard 标准化
        train_set,val_set=gcn_pickfold(featureList, train_index, test_index)

        net=Net(dim=256)
        
        net.to(args.device)

        train_dataset = MyGCNDataset(train_set)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = MyGCNDataset(val_set)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = MyGCNDataset(testlist)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        criterion = torch.nn.MSELoss()
        # criterion = torch.nn.HuberLoss(delta=args.alpha)
        optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-2)
        writer = SummaryWriter('/mnt/data/xukeyu/PPA_Pred/tmp/log/'+TIMESTAMP+'val'+str(i))
        
        for epoch in range(args.epoch):
            #train
            net, train_prelist, train_truelist, train_loss = cnn_train(net, train_dataloader, optimizer, criterion, args)
            df = pd.DataFrame({'label':train_truelist, 'pre':train_prelist})
            train_pcc = df.pre.corr(df.label)
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/pcc', train_pcc, epoch)
            logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss))
            
            #val
            names,val_prelist, val_truelist,val_loss = cnn_predict(net, val_dataloader, criterion, args)
            df = pd.DataFrame({'label':val_truelist, 'pre':val_prelist})
            val_pcc = df.pre.corr(df.label)
            val_mae = F.l1_loss(torch.tensor(val_prelist),torch.tensor(val_truelist))
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/pcc', val_pcc, epoch)
            logging.info("Epoch "+ str(epoch)+ ": val Loss = %.4f"%(val_loss)+ ", val mae = %.4f"%(val_mae))
            
            #test
            names,test_prelist, test_truelist,_ = cnn_predict(net, test_dataloader, criterion, args)
            df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
            test_pcc = df.pre.corr(df.label)
            test_mae = F.l1_loss(torch.tensor(test_prelist),torch.tensor(test_truelist))
            
            if math.fabs(test_pcc) > best_test_pcc[i]:
                best_pcc[i]=math.fabs(val_pcc)
                best_epoch[i] = epoch
                best_mae[i] = val_mae
                best_test_pcc[i] = test_pcc
                best_test_mae[i] = test_mae
                torch.save(net.state_dict(),f'{args.modeldir}affinity_ban{i}.pt')
                with open(f'{args.outdir}ban_pred/result_{i}.txt','w') as f:
                    for j in range(0,len(test_truelist)):
                        f.write(names[j])
                        f.write('\t')
                        f.write(str(test_truelist[j]))
                        f.write('\t')
                        f.write(str(test_prelist[j]))
                        f.write('\n')
    
    pcc=0.
    mae=0.
    pcc1 = 0.
    mae1 = 0.
    for i in range(5):
        pcc=pcc+best_pcc[i]
        mae=mae+best_mae[i]
        pcc1=pcc1+best_test_pcc[i]
        mae1=mae1+best_test_mae[i]
        logging.info('val_'+str(i)+' best_pcc = %.4f'%(best_pcc[i])+' , best_mae = %.4f'%(best_mae[i])+' , best_epoch : '+str(best_epoch[i]))
    print('pcc  :   '+str(pcc/5))
    print('mae  :   '+str(mae/5))
    print('pcc1  :   '+str(pcc1/5))
    print('mae1  :   '+str(mae1/5))
            
