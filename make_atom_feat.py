import os
import torch
import pickle
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import pandas as pd
from utils.parse_args import get_args
from utils.generateGraph import generate_residue_graph

if __name__ == '__main__':
    args=get_args()
    print(args)

    complexdict={} # pdbname : [seq1, seq2, bingding_affinity]
    
    files=os.listdir("../contact_graph/")
    pdb_atom_graph=set()
    for file in files:
        pdbname=file.split('_')[0]
        pdb_atom_graph.add(pdbname)
    files=os.listdir("../isomorphic/refined_feat/")

    for file in files:
        pdb_chain=file.split('.')[0]
        pdbname=pdb_chain.split('_')[0]
        if pdbname not in pdb_atom_graph:
            continue
        if pdbname not in complexdict.keys():
            complexdict[pdbname]=[]
        complexdict[pdbname].append(pdb_chain)
        
    maxlen=0
    for pdbname in complexdict.keys():
        # if pdbname != '1lhu':   continue
        # local redisue
        with open('../contact_inside_graph/'+pdbname+'_connect.picke', 'rb') as file:
            connect = pickle.load(file)
        node_feature={}
        logging.info("generate graph :"+pdbname)
        for pdb_chains in complexdict[pdbname]:
            with open('../isomorphic/refined_feat/'+pdb_chains+'.picke', 'rb') as file:
                atom_feat = pickle.load(file)
            chain=pdb_chains.split('_')[1]
            for k,v in atom_feat.items():
                feat=[]
                feat.append(v[0])
                feat.append([v[1]])
                feat.append(v[2])
                feat.append(v[3])
                feat.append([float(v[4]),v[5],v[6],v[7],v[8],v[9],v[10]])
                node_feature[chain+'_'+str(k)]=feat
        node_features, edge_index,edge_attr=generate_residue_graph(pdbname,node_feature,connect,args.padding)
        # print(node_features)
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
        edge_attr=torch.tensor(edge_attr,dtype=torch.float)
        torch.save(x.to(torch.device('cpu')),'./data/atom_graph/ligand/'+pdbname+"_x"+'.pth')
        torch.save(edge_index.to(torch.device('cpu')),'./data/atom_graph/ligand/'+pdbname+"edge_index"+'.pth')
        torch.save(edge_attr.to(torch.device('cpu')),'./data/atom_graph/ligand/'+pdbname+"_edge_attr"+'.pth')