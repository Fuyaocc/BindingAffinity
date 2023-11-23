import os
import re
import torch
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,format='%(levelname)s: %(message)s')
import pandas as pd
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
    
    general_set_complex_dict={}
    with open('./data/general-set.txt', 'r') as f:
        for line in f:
            pkd=line[19:23]
            general_name=line[0:4]
            general_set_complex_dict[general_name]=float(pkd)
        
    pp_complex_dict={}
    with open('./data/pdbbind_data.txt', 'r') as f:
        for line in f:
            blocks=re.split('\t|\n',line)
            pdbname=blocks[0]
            pp_complex_dict[pdbname]=float(blocks[1])

    pp_complex_list=set()
    ligand_complex_list=set()
    general_set_complex_list=set()
    
    pps=os.listdir('./data/atom_graph/pp/')
    ligands=os.listdir('./data/atom_graph/ligand/')
    generals = os.listdir('./data/atom_graph/general-set/')
    
    for pp in pps:
        if pp[:4] in pp_complex_dict.keys():
            pp_complex_list.add(pp[:4])
    
    for ligand in ligands:
        if ligand[:4] in ligand_complex_dict.keys():
            ligand_complex_list.add(ligand[:4])

    for general in generals:
        if general[:4] in general_set_complex_dict.keys():
            general_set_complex_list.add(general[:4])
        
    test_complexdict=set()
    for line in open(args.inputDir+'test_set.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        test_complexdict.add(pdbname)
        
    featureList=[]
    labelList=[]

    test_featureList=[]
    test_labelList=[]
    # for pdbname in pp_complex_list:#local redisue
    #     logging.info("load pp data graph :"+pdbname)
    #     x = torch.load('./data/atom_graph/pp/'+pdbname+"_x"+'.pth').to(torch.float)
    #     edge_index=torch.load('./data/atom_graph/pp/'+pdbname+"edge_index"+'.pth').to(torch.int64)
    #     edge_attr=torch.load('./data/atom_graph/pp/'+pdbname+"_edge_attr"+'.pth').to(torch.float)
    #     if(x.shape[0]==0 or edge_index.shape[0]==0 or edge_attr.shape[0]==0 ):
    #         continue
    #     data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=pp_complex_dict[pdbname],distillation_y=[0],name=pdbname,energy=[0])
    #     if pdbname in test_complexdict:
    #         test_featureList.append(data)
    #         test_labelList.append(pp_complex_dict[pdbname])
    #     else:
    #         featureList.append(data)
    #         labelList.append(pp_complex_dict[pdbname])
    torch.set_printoptions(profile="full")
    for pdbname in ligand_complex_list:
        logging.info("load ligand data graph :"+pdbname)
        x = torch.load('./data/atom_graph/ligand/'+pdbname+"_x"+'.pth').to(torch.float)
        edge_index=torch.load('./data/atom_graph/ligand/'+pdbname+"edge_index"+'.pth').to(torch.int64)
        edge_attr=torch.load('./data/atom_graph/ligand/'+pdbname+"_edge_attr"+'.pth').to(torch.float)
        if(x.shape[0]==0 or edge_index.shape[0]==0 or edge_attr.shape[0]==0 ):
            continue
        if torch.isnan(x).any():
            continue

        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=ligand_complex_dict[pdbname],distillation_y=[0],name=pdbname,energy=[0])
        featureList.append(data)
        labelList.append(ligand_complex_dict[pdbname])
    
    for pdbname in general_set_complex_list:
        logging.info("load general-set data graph :"+pdbname)
        x = torch.load('./data/atom_graph/general-set/'+pdbname+"_x"+'.pth').to(torch.float)
        edge_index=torch.load('./data/atom_graph/general-set/'+pdbname+"edge_index"+'.pth').to(torch.int64)
        edge_attr=torch.load('./data/atom_graph/general-set/'+pdbname+"_edge_attr"+'.pth').to(torch.float)
        if(x.shape[0]==0 or edge_index.shape[0]==0 or edge_attr.shape[0]==0 ):
            continue
        if torch.isnan(x).any():
            continue
        
        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=general_set_complex_dict[pdbname],distillation_y=[0],name=pdbname,energy=[0])
        featureList.append(data)
        labelList.append(general_set_complex_dict[pdbname])

    logging.info(len(featureList))
    #交叉验证
    kf = KFold(n_splits=5,random_state=42, shuffle=True)
    best_pcc=[0.0,0.0,0.0,0.0,0.0]
    best_epoch=[0,0,0,0,0,0,0,0,0,0]
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        net=Net(input_dim=args.dim,hidden_dim=64,output_dim=64)
        net.to(args.device)

        train_set,val_set=gcn_pickfold(featureList, train_index, test_index)
        train_dataset=MyGCNDataset(train_set)
        train_dataloader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

        val_dataset=MyGCNDataset(val_set)
        val_dataloader=DataLoader(val_dataset, batch_size=args.batch_size//4, shuffle=True, num_workers=1, pin_memory=True)

        criterion = torch.nn.MSELoss()
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-4)

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
                torch.save(net.state_dict(),f'./models/saved/gcn/affinity_model{i}_dim{args.dim}.pt')
            
            #test
            # test_dataset=MyGCNDataset(test_featureList)
            # test_dataloader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)
            # names,test_prelist, test_truelist,test_loss = gcn_predict(net,test_dataloader,criterion,args.device,i,0)
            # df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
            # test_pcc = df.pre.corr(df.label)
            # writer.add_scalar('affinity_val/test_loss', test_loss, epoch)
            # writer.add_scalar('affinity_val/test_pcc', test_pcc, epoch)
            # logging.info("Epoch "+ str(epoch)+ ": Test Loss = %.4f"%(test_loss))
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
        logging.info('val_'+str(i)+' best_pcc = %.4f'%(best_pcc[i]) +' , best_epoch : '+str(best_epoch[i]))
    
    print('pcc  :   '+str(pcc/5))