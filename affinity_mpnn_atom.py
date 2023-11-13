import os
import re
import torch
import math
import pickle
import numpy as np
import sys
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader,Data,Batch
from utils.parse_args import get_args
from utils.seq2ESM1v import seq2ESM1v
from utils.MyDataset import MyGCNDataset,gcn_pickfold
from utils.run_epoch import mpnn_train,gcn_predict
from utils.generateGraph import generate_residue_graph
from utils.resFeature import getAAOneHotPhys
from utils.getSASA import getDSSP
from utils.readFoldX import readFoldXResult
from utils.getInterfaceRate import getInterfaceRateAndSeq
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
    
    pp_complex_dict={}
    with open('./data/pdbbind_data.txt', 'r') as f:
        for line in f:
            blocks=re.split('\t|\n',line)
            pdbname=blocks[0]
            pp_complex_dict[pdbname]=float(blocks[1])

    pp_complex_list=set()
    ligand_complex_list=set()
    
    pps=os.listdir('./data/atom_graph/pp/')
    ligands=os.listdir('./data/atom_graph/ligand/')
    
    for pp in pps:
        if pp[:4] in pp_complex_dict.keys():
            pp_complex_list.add(pp[:4])
    
    for ligand in ligands:
        if ligand[:4] in ligand_complex_dict.keys():
            ligand_complex_list.add(ligand[:4])
        
    
    
    test_complexdict=set()
    for line in open(args.inputDir+'test_set.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        test_complexdict.add(pdbname)
        
    featureList=[]
    labelList=[]

    test_featureList=[]
    test_labelList=[]
    for pdbname in pp_complex_list:
        #local redisue
        logging.info("load pp data graph :"+pdbname)
        x = torch.load('./data/atom_graph/pp/'+pdbname+"_x"+'.pth').to(torch.float).to(args.device)
        edge_index=torch.load('./data/atom_graph/pp/'+pdbname+"edge_index"+'.pth').to(torch.int64).to(args.device)
        edge_attr=torch.load('./data/atom_graph/pp/'+pdbname+"_edge_attr"+'.pth').to(torch.float).to(args.device)
        if(x.shape[0]==0 or edge_index.shape[0]==0 or edge_attr.shape[0]==0 ):
            continue
        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=pp_complex_dict[pdbname],distillation_y=[0],name=pdbname,energy=[0])
        if pdbname in test_complexdict:
            test_featureList.append(data)
            test_labelList.append(pp_complex_dict[pdbname])
        else:
            featureList.append(data)
            labelList.append(pp_complex_dict[pdbname])
    
    for pdbname in ligand_complex_list:
        logging.info("load ligand data graph :"+pdbname)
        x = torch.load('./data/atom_graph/ligand/'+pdbname+"_x"+'.pth').to(torch.float).to(args.device)
        edge_index=torch.load('./data/atom_graph/ligand/'+pdbname+"edge_index"+'.pth').to(torch.int64).to(args.device)
        edge_attr=torch.load('./data/atom_graph/ligand/'+pdbname+"_edge_attr"+'.pth').to(torch.float).to(args.device)
        if(x.shape[0]==0 or edge_index.shape[0]==0 or edge_attr.shape[0]==0 ):
            continue
        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=ligand_complex_dict[pdbname],distillation_y=[0],name=pdbname,energy=[0])
        featureList.append(data)
        labelList.append(ligand_complex_dict[pdbname])

    #交叉验证
    kf = KFold(n_splits=5,random_state=42, shuffle=True)
    best_pcc=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mse=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_rmse=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mae=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_epoch=[0,0,0,0,0,0,0,0,0,0]
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        net=Net(input_dim=args.dim
                        ,hidden_dim=16
                        ,output_dim=16)
        
        net.to(args.device)

        train_set,val_set=gcn_pickfold(featureList, train_index, test_index)
        train_dataset=MyGCNDataset(train_set)
        print(len(train_dataset))
        train_dataloader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)

        criterion = torch.nn.MSELoss()
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        optimizer = torch.optim.Adam(net.parameters(), lr = 1e-2, weight_decay = 1e-3)

        writer = SummaryWriter(args.logdir+str(i))
        
        for epoch in range(args.epoch):
            #train
            net,train_prelist, train_truelist, train_loss= mpnn_train(net
                                                                    ,train_dataloader
                                                                    ,optimizer
                                                                    ,criterion
                                                                    ,args.device
                                                                    ,i
                                                                    ,epoch
                                                                    ,args.outDir
                                                                    ,args.epsilon
                                                                    ,args.alpha
                                                                    ,kl_loss)
                                                                                                   
            logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss))

            df = pd.DataFrame({'label':train_truelist, 'pre':train_prelist})
            train_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_train/loss', train_loss, epoch)
            writer.add_scalar('affinity_train/pcc', train_pcc, epoch)
            
            #val
            val_dataset=MyGCNDataset(val_set)
            val_dataloader=DataLoader(val_dataset, batch_size=32, shuffle=True)
            _,val_prelist, val_truelist,val_loss = gcn_predict(net, val_dataloader, criterion, args.device, i, epoch)
            df = pd.DataFrame({'label':val_truelist, 'pre':val_prelist})
            val_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_val/val_loss', val_loss, epoch)
            writer.add_scalar('affinity_val/val_pcc', val_pcc, epoch)
            mae=F.l1_loss(torch.tensor(val_prelist), torch.tensor(val_truelist))
            mse=F.mse_loss(torch.tensor(val_prelist), torch.tensor(val_truelist))
            rmse=math.sqrt(mse)
            logging.info("Epoch "+ str(epoch)+ ": val Loss = %.4f"%(val_loss)+" , mse = %.4f"%(mse)+" , rmse = %.4f"%(rmse)+" , mae = %.4f"%(mae))
            if rmse < 2.78 and val_pcc > best_pcc[i]:
                best_pcc[i]=val_pcc
                best_mse[i]=mse
                best_rmse[i]=rmse
                best_mae[i]=mae
                best_epoch[i]=epoch
                torch.save(net.state_dict(),f'./models/saved/gcn/affinity_model{i}_dim{args.dim}_foldx.pt')
            
            #test
            test_dataset=MyGCNDataset(test_featureList)
            test_dataloader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)
            names,test_prelist, test_truelist,test_loss = gcn_predict(net,test_dataloader,criterion,args.device,i,0)
            df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
            mae=F.l1_loss(torch.tensor(test_prelist),torch.tensor(test_truelist))
            mse=F.mse_loss(torch.tensor(test_prelist),torch.tensor(test_truelist))
            rmse=math.sqrt(mse)
            test_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_val/test_loss', test_loss, epoch)
            writer.add_scalar('affinity_val/test_pcc', test_pcc, epoch)
            logging.info("Epoch "+ str(epoch)+ ": Test Loss = %.4f"%(test_loss)+" , mse = %.4f"%(mse)+" , pcc = %.4f"%(test_pcc))
            # with open(f'./tmp/pred/result_{i}.txt','w') as f:
            #     for j in range(0,len(test_truelist)):
            #         f.write(names[j])
            #         f.write('\t')
            #         f.write(str(test_prelist[j]))
            #         f.write('\t')
            #         f.write(str(test_truelist[j]))
            #         f.write('\n')
    
    pcc=0.
    mse=0.
    rmse=0.
    mae=0.
    for i in range(5):
        pcc=pcc+best_pcc[i]
        mse=mse+best_mse[i]
        rmse=rmse+best_rmse[i]
        mae=mae+best_mae[i]
        logging.info('val_'+str(i)+' best_pcc = %.4f'%(best_pcc[i])+' , best_mse = %.4f'%(best_mse[i])+' , best_rmse = %.4f'%(best_rmse[i])+' , best_mae = %.4f'%(best_mae[i])+' , best_epoch : '+str(best_epoch[i]))

    
    print('pcc  :   '+str(pcc/5))
    print('mse  :   '+str(mse/5))
    print('rmse :   '+str(rmse/5))
    print('mae  :   '+str(mae/5))
            