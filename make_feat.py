import os
import re
import torch
import math
import numpy as np
import pickle
import sys
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import esm
import pandas as pd
import torch.nn.functional as F
import torch.optim.lr_scheduler as ls
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader,Data
from sklearn.decomposition import PCA
from utils.parse_args import get_args
from utils.seq2ESM1v import seq2ESM1v
from utils.MyDataset import MyGCNDataset,gcn_pickfold
from utils.generateGraph import generate_residue_graph
from utils.resFeature import getAAOneHotPhys
from utils.getSASA import getDSSP
from utils.readFoldX import readFoldXResult
from utils.getInterfaceRate import getInterfaceRateAndSeq
from models.affinity_net_gcn import AffinityNet

if __name__ == '__main__':
    args=get_args()
    print(args)

    complexdict={} # pdbname : [seq1, seq2, bingding_affinity]

    for line in open(args.inputDir+'pdbbind_data.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        complexdict[pdbname]=float(blocks[1])
    
    
    filter_set=set(["1de4","1ogy","1tzn","2wss","3lk4","3sua","3swp","4r8i","6c74","4nzl","3l33","4dvg"])#data in test_set or dirty_data
    too_long=set(['3nog', '4bxz', '3pvm', '4ksd', '3kls', '3noc','2nz9', '2j8s', '2nyy','5w1t', '5i5k', '6eyd', '5hxb'])
    # files=os.listdir("./data/graph/")
    graph_dict=set()
    # for file in files:
    #     graph_dict.add(file.split("_")[0])
        
    resfeat=getAAOneHotPhys()

    featureList=[]
    labelList=[]
    i=0
    maxlen=0
    
    dirty_data=set()
    maxchians=0
    maxchains_name=''
    xx=False
    for pdbname in complexdict.keys():
        # if pdbname != '4fzv':   continue
        # if pdbname == '2j8s':   xx=True
        # if xx==False:   continue
        if pdbname in filter_set or pdbname in too_long:continue
        #local redisue
        if pdbname.startswith("5ma"):continue #dssp can't cal rSA
        if True:
            pdb_path='./data/pdbs/'+pdbname+'.pdb'
            # energy=readFoldXResult(args.foldxPath,pdbname)
            # energy=torch.tensor(energy,dtype=torch.float)
            # seq,interfaceDict,chainlist,connect=getInterfaceRateAndSeq(pdb_path,interfaceDis=args.interfacedis)
            with open('./data/graphfeat/'+pdbname+'_interfaceDict.picke', 'rb') as file:
                interfaceDict = pickle.load(file)
            with open('./data/graphfeat/'+pdbname+'_connect.picke', 'rb') as file:
                connect = pickle.load(file)
            with open('./data/graphfeat/'+pdbname+'_seq.picke', 'rb') as file:
                seq = pickle.load(file)
            chainlist=interfaceDict.keys()
            dssp=getDSSP(pdb_path)
            node_feature={}
            logging.info("generate graph :"+pdbname)
            flag=False
            chain_sign=0
            print(chainlist)
            for chain in chainlist:
                if flag==True:break
                with open('../sidechain/'+pdbname+'_'+chain+'_sidechain.picke', 'rb') as file:
                    sidechain = pickle.load(file)
                with open('../sidechain/'+pdbname+'_'+chain+'_sidechain_center.picke', 'rb') as file:
                   sidechain_center = pickle.load(file)
                reslist=interfaceDict[chain]
                esm1f_feat=torch.load('./data/esmfeature/struct_emb/'+pdbname+'_'+chain+'.pth')
                esm1f_feat=F.avg_pool1d(esm1f_feat,16,16)
                esm1v_feat=torch.load('./data/esmfeature/seq_emb/'+pdbname+'_'+chain+'.pth')
                esm1v_feat=F.avg_pool1d(esm1v_feat,40,40)
                for v in reslist:
                    reduise=v.split('_')[1]
                    s = seq[pdbname+'_'+chain][1]
                    index=int(reduise[1:])-int(s)
                    if (chain,index+1) not in dssp.keys(): 
                        other_feat=[0.0]
                    else:          
                        other_feat=[dssp[(chain,index+1)][3]]#[rSA,...]
                    node_feature[v]=[]
                    node_feature[v].append([chain_sign])
                    node_feature[v].append(sidechain_center[index])
                    node_feature[v].append(resfeat[reduise[0]])
                    node_feature[v].append(other_feat)
                    node_feature[v].append(sidechain[index])
                    node_feature[v].append(esm1f_feat[index].tolist())
                    node_feature[v].append(esm1v_feat[index].tolist())
                chain_sign+=1
            node_features, edge_index,edge_attr=generate_residue_graph(pdbname,node_feature,connect,args.padding)
            x = torch.tensor(node_features, dtype=torch.float)
            print(x.shape)
            edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
            edge_attr=torch.tensor(edge_attr,dtype=torch.float)
            torch.save(x.to(torch.device('cpu')),'./data/graph/'+pdbname+"_chain_sign"+'.pth')
            torch.save(edge_index.to(torch.device('cpu')),'./data/graphfeat/'+pdbname+"_edge_index"+'.pth')
            torch.save(edge_attr.to(torch.device('cpu')),'./data/graphfeat/'+pdbname+"_edge_attr"+'.pth')
    #         torch.save(energy.to(torch.device('cpu')),'./data/skempi/graphfeat/'+pdbname+"_energy"+'.pth')      
    #     data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=complexdict[pdbname],name=pdbname,energy=energy)
    #     maxlen+=1
    #     featureList.append(data)
    #     labelList.append(complexdict[pdbname])
    # logging.info(maxlen)
    # #交叉验证
    # kf = KFold(n_splits=5,random_state=2022, shuffle=True)
    # best_pcc=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    # best_mse=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    # best_rmse=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    # best_mae=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    # best_epoch=[0,0,0,0,0,0,0,0,0,0]
    # for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
    #     net=AffinityNet(in_channels=args.dim
    #                                ,hidden_channels=args.dim
    #                                ,out_channels=args.dim
    #                                ,mlp_in_channels=args.dim+21
    #                                ,device=args.device)
    #     net.to(args.device)

    #     train_set,val_set=gcn_pickfold(featureList,train_index,test_index)
    #     # train_set += skempi_featureList
        
    #     train_dataset=MyGCNDataset(train_set)
    #     train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

    #     val_dataset=MyGCNDataset(val_set)
    #     val_dataloader=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True)

    #     criterion = torch.nn.MSELoss()
    #     optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-3)

    #     writer = SummaryWriter('./log/val'+str(i))
        
    #     for epoch in range(args.epoch):
    #         #train
    #         # net,train_prelist, train_truelist, train_loss,normal_mse,against_mse,against_js= gcn_train(net,train_dataloader,optimizer,criterion,args.device,i,epoch,args.outDir,args.epsilon,args.alpha)
    #         net,train_prelist, train_truelist, train_loss = gcn_train(net,train_dataloader,optimizer,criterion,args.device,i,epoch,args.outDir,args.epsilon,args.alpha)
                                                                                                   
    #         # logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss)+" ,mse = %.4f"%(normal_mse)+" ,against_mse= %.4f"%(against_mse)+" ,against_js = %.4f"%(against_js))
    #         logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss))

    #         df = pd.DataFrame({'label':train_truelist, 'pre':train_prelist})
    #         train_pcc = df.pre.corr(df.label)
    #         writer.add_scalar('affinity_train/loss', train_loss, epoch)
    #         # writer.add_scalar('affinity_train/normal_mse', normal_mse, epoch)
    #         # writer.add_scalar('affinity_train/against_mse', against_mse, epoch)
    #         # writer.add_scalar('affinity_train/against_js', against_js, epoch)
    #         writer.add_scalar('affinity_train/pcc', train_pcc, epoch)
            
    #         #val
    #         val_prelist, val_truelist,val_loss = gcn_predict(net,val_dataloader,criterion,args.device,i,epoch)
    #         df = pd.DataFrame({'label':val_truelist, 'pre':val_prelist})
    #         val_pcc = df.pre.corr(df.label)
    #         writer.add_scalar('affinity_val/loss', val_loss, epoch)
    #         writer.add_scalar('affinity_val/pcc', val_pcc, epoch)
    #         mae=F.l1_loss(torch.tensor(val_prelist),torch.tensor(val_truelist))
    #         mse=F.mse_loss(torch.tensor(val_prelist),torch.tensor(val_truelist))
    #         rmse=math.sqrt(mse)
    #         logging.info("Epoch "+ str(epoch)+ ": val Loss = %.4f"%(val_loss)+" ,mse = %.4f"%(mse)+" ,rmse = %.4f"%(rmse)+" ,mae = %.4f"%(mae))
    #         if rmse < 2.78 and val_pcc > best_pcc[i]:
    #             best_pcc[i]=val_pcc
    #             best_mse[i]=mse
    #             best_rmse[i]=rmse
    #             best_mae[i]=mae
    #             best_epoch[i]=epoch
    #             torch.save(net.state_dict(),f'./models/saved/gcn/affinity_model{i}_dim{args.dim}_foldx.pt')
    
    # pcc=0.
    # mse=0.
    # rmse=0.
    # mae=0.
    # for i in range(5):
    #     pcc=pcc+best_pcc[i]
    #     mse=mse+best_mse[i]
    #     rmse=rmse+best_rmse[i]
    #     mae=mae+best_mae[i]
    #     logging.info('val_'+str(i)+' best_pcc = %.4f'%(best_pcc[i])+' , best_mse = %.4f'%(best_mse[i])+' , best_rmse = %.4f'%(best_rmse[i])+' , best_mae = %.4f'%(best_mae[i])+' , best_epoch : '+str(best_epoch[i]))

    
    # print('pcc  :   '+str(pcc/5))
    # print('mse  :   '+str(mse/5))
    # print('rmse :   '+str(rmse/5))
    # print('mae  :   '+str(mae/5))
            