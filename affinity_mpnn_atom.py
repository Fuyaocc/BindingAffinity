import os
import re
import torch
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils.parse_args import get_args
from utils.MyDataset import MyGCNDataset,gcn_pickfold
from utils.run_epoch import mpnn_train,gcn_predict
from models.affinity_net_mpnn_atom import Net
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

    scaler_zscore = StandardScaler()
    for pdbname in ligand_complex_list:
        logging.info("load ligand data graph :"+pdbname)
        x = torch.load('./data/atom_graph/ligand/'+pdbname+"_x"+'.pth').to(torch.float32)
        edge_index=torch.load('./data/atom_graph/ligand/'+pdbname+"edge_index"+'.pth').to(torch.int32)
        edge_attr=torch.load('./data/atom_graph/ligand/'+pdbname+"_edge_attr"+'.pth').to(torch.float32)
        if(x.shape[0]==0 or edge_index.shape[0]==0 or edge_attr.shape[0]==0 ):
            continue
        idx = torch.isnan(x)
        x[idx] = float(0.0)
        idx = torch.isinf(x)
        x[idx] = float(0.0)
        x = scaler_zscore.fit_transform(x.numpy())
        x = torch.tensor(x).to(torch.float32)
        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=ligand_complex_dict[pdbname],distillation_y=[0],name=pdbname,energy=[0])
        featureList.append(data)
        labelList.append(ligand_complex_dict[pdbname])
    
    for pdbname in general_set_complex_list:
        logging.info("load general-set data graph :"+pdbname)
        x = torch.load('./data/atom_graph/general-set/'+pdbname+"_x"+'.pth').to(torch.float32)
        edge_index=torch.load('./data/atom_graph/general-set/'+pdbname+"edge_index"+'.pth').to(torch.int32)
        edge_attr=torch.load('./data/atom_graph/general-set/'+pdbname+"_edge_attr"+'.pth').to(torch.float32)
        if(x.shape[0]==0 or edge_index.shape[0]==0 or edge_attr.shape[0]==0 ):
            continue
        idx = torch.isnan(x)
        x[idx] = float(0.0)
        idx = torch.isinf(x)
        x[idx] = float(0.0)
        x = scaler_zscore.fit_transform(x.numpy())
        x = torch.tensor(x).to(torch.float32)
        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=general_set_complex_dict[pdbname],distillation_y=[0],name=pdbname,energy=[0])
        featureList.append(data)
        labelList.append(general_set_complex_dict[pdbname])

    logging.info(len(featureList))
    #交叉验证
    kf = KFold(n_splits=5,random_state=42, shuffle=True)
    best_pcc=[0.0,0.0,0.0,0.0,0.0]
    best_epoch=[0,0,0,0,0,0,0,0,0,0]
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        net=Net(input_dim=args.dim,hidden_dim=64,output_dim=32)
        net.to(args.device)

        train_set,val_set=gcn_pickfold(featureList, train_index, test_index)
        train_dataset=MyGCNDataset(train_set)
        train_dataloader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

        val_dataset=MyGCNDataset(val_set)
        val_dataloader=DataLoader(val_dataset, batch_size=args.batch_size//4, shuffle=True, num_workers=1, pin_memory=True)

        criterion = torch.nn.MSELoss()
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, weight_decay = 0.002)

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
    
    pcc=0.
    mse=0.
    for i in range(5):
        pcc=pcc+best_pcc[i]
        logging.info('val_'+str(i)+' best_pcc = %.4f'%(best_pcc[i]) +' , best_epoch : '+str(best_epoch[i]))
    print('pcc  :   '+str(pcc/5))