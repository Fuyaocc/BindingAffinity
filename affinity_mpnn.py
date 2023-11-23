import os
import re
import torch
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader,Data,Batch
from utils.parse_args import get_args
from utils.MyDataset import MyGCNDataset,gcn_pickfold
from utils.run_epoch import mpnn_train,gcn_predict
from utils.resFeature import getAAOneHotPhys
from models.affinity_net_mpnn import Net


def collate_fn(data_list):
    return Batch.from_data_list(data_list)

if __name__ == '__main__':
    args=get_args()
    print(args)

    complexdict={} # pdbname : [seq1, seq2, bingding_affinity]

    for line in open(args.inputDir+'pdbbind_data.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        complexdict[pdbname]=float(blocks[1])
    
    test_complexdict=set()
    for line in open(args.inputDir+'test_set.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        test_complexdict.add(pdbname)
    
    distillation_data = {}
    with open(args.inputDir+'distillation_data.txt') as f:
        for line in f:
            v = re.split("\t|\n",line)
            distillation_data[v[0]]=float(v[1])
    
    filter_set=set(["1de4","1ogy","1tzn","2wss","3lk4","3sua","3swp","4r8i","6c74","4nzl","3l33","4dvg"])#data in test_set or dirty_data
    too_long=set(['3nog', '4bxz', '3pvm', '4ksd', '3kls', '3noc','2nz9', '2j8s', '2nyy','5w1t', '5i5k', '6eyd', '5hxb'])
    files=os.listdir("./data/graph/")
    graph_dict=set()
    for file in files:
        graph_dict.add(file.split("_")[0])
        
    resfeat=getAAOneHotPhys()

    featureList=[]
    labelList=[]
    test_featureList=[]
    test_labelList=[]
    for pdbname in complexdict.keys():
        if pdbname in filter_set or pdbname in too_long:continue 
        #local redisue
        if pdbname.startswith("5ma"):continue #dssp can't cal rSA
        if pdbname in graph_dict:
            logging.info("load pdbbind data graph :"+pdbname)
            x = torch.load('./data/graph/'+pdbname+"_chain_sign"+'.pth').to(torch.float).to(args.device)
            edge_index=torch.load('./data/graphfeat/'+pdbname+"edge_index"+'.pth').to(torch.int64).to(args.device)
            edge_attr=torch.load('./data/graphfeat/'+pdbname+"_edge_attr"+'.pth').to(torch.float).to(args.device)
            energy=torch.load('./data/graph/'+pdbname+"_energy"+'.pth').to(torch.float).to(args.device)
            # print(x.shape)
            idx=torch.isnan(x)
            x[idx]=0.0

            energy = energy[:21]
            data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=complexdict[pdbname],distillation_y=distillation_data[pdbname],name=pdbname,energy=energy)

            if pdbname in test_complexdict:
                test_featureList.append(data)
                test_labelList.append(complexdict[pdbname])
            else:
                featureList.append(data)
                labelList.append(complexdict[pdbname])

    logging.info(len(featureList))
    #交叉验证
    kf = KFold(n_splits=5,random_state=2023, shuffle=True)
    best_pcc=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mse=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_rmse=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mae=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_epoch=[0,0,0,0,0,0,0,0,0,0]
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        net=Net(input_dim=args.dim
                        ,hidden_dim=64
                        ,output_dim=32)
        
        net.to(args.device)

        train_set,val_set=gcn_pickfold(featureList, train_index, test_index)
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
            if epoch % 100 == 0 and val_pcc > best_pcc[i]:
                best_pcc[i]=val_pcc
                best_epoch[i]=epoch
                torch.save(net.state_dict(),f'./models/saved/gcn/affinity_model{i}_dim{args.dim}_foldx.pt')
            
            #test
            # test_dataset=MyGCNDataset(test_featureList)
            # test_dataloader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)
            # names,test_prelist, test_truelist,test_loss = gcn_predict(net,test_dataloader,criterion,args.device,i,0)
            # df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
            # mae=F.l1_loss(torch.tensor(test_prelist),torch.tensor(test_truelist))
            # mse=F.mse_loss(torch.tensor(test_prelist),torch.tensor(test_truelist))
            # rmse=math.sqrt(mse)
            # test_pcc = df.pre.corr(df.label)
            # writer.add_scalar('affinity_val/test_loss', test_loss, epoch)
            # writer.add_scalar('affinity_val/test_pcc', test_pcc, epoch)
            # logging.info("Epoch "+ str(epoch)+ ": Test Loss = %.4f"%(test_loss)+" , mse = %.4f"%(mse)+" , pcc = %.4f"%(test_pcc))
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
            