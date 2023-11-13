import os
import re
import torch
import math
import numpy as np
import pickle
import sys
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from utils.parse_args import get_args
from utils.generateGraph import generate_residue_graph
from utils.resFeature import getAAOneHotPhys
from utils.getInterfaceRate import getInterfaceRateAndSeq

if __name__ == '__main__':
    args=get_args()
    print(args)

    complexdict={} # pdbname : [seq1, seq2, bingding_affinity]
    
    
    files=os.listdir("/home/ysbgs/xky/contact_graph/")
    pdb_atom_graph=set()
    for file in files:
        pdbname=file.split('_')[0]
        pdb_atom_graph.add(pdbname)
    files=os.listdir("/home/ysbgs/xky/ligand_feat/")
    for file in files:
        pdb_chain=file.split('.')[0]
        pdbname=pdb_chain.split('_')[0]
        if pdbname not in pdb_atom_graph:
            continue
        if pdbname not in complexdict.keys():
            complexdict[pdbname]=[]
        complexdict[pdbname].append(pdb_chain)
        
    resfeat=getAAOneHotPhys()

    maxlen=0
    le = LabelEncoder()
    atom_sysmol=['H','Br', 'Cs', 'Se', 'N', 'Cd', 'O', 'Ni', 'Sr', 'I', 'S', 'Mn', 'K', 'Na', 'C', 'Cu', 'Ca', 'Cl', 'P', 'Co', 'Hg', 'Fe', 'Zn', 'Mg', 'F']
    le.fit(atom_sysmol)
    for pdbname in complexdict.keys():
        # if pdbname != '4fzv':   continue
        #local redisue
        with open('/home/ysbgs/xky/contact_inside_graph/'+pdbname+'_connect.picke', 'rb') as file:
            connect = pickle.load(file)
        node_feature={}
        chain_sign=0
        logging.info("generate graph :"+pdbname)
        for pdb_chains in complexdict[pdbname]:
            with open('/home/ysbgs/xky/ligand_feat/'+pdb_chains+'.picke', 'rb') as file:
                atom_feat = pickle.load(file)
            chain=pdb_chains.split('_')[1]
            for k,v in atom_feat.items():
                feat=[]
                feat.append([le.transform([v[0]])[0]])
                feat.append([v[2]])
                feat.append([v[4]])
                if(v[5]==True):
                    feat.append([1.])
                else:
                    feat.append([0.])
                if(v[7]==True):
                    feat.append([1.])
                else:
                    feat.append([0.])
                feat.append(v[8])
                node_feature[chain+'_'+str(k)]=feat
        # print(node_feature)
        node_features, edge_index,edge_attr=generate_residue_graph(pdbname,node_feature,connect,args.padding)
        # print(node_features)
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
        edge_attr=torch.tensor(edge_attr,dtype=torch.float)
        torch.save(x.to(torch.device('cpu')),'./data/atom_graph/ligand/'+pdbname+"_x"+'.pth')
        torch.save(edge_index.to(torch.device('cpu')),'./data/atom_graph/ligand/'+pdbname+"edge_index"+'.pth')
        torch.save(edge_attr.to(torch.device('cpu')),'./data/atom_graph/ligand/'+pdbname+"_edge_attr"+'.pth')
#         torch.save(energy.to(torch.device('cpu')),'./data/skempi/graphfeat/'+pdbname+"_energy"+'.pth')      
