import os
import re
import torch
import pickle
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import torch.nn.functional as F
import numpy as np
from utils.parse_args import get_args
from utils.getInterfaceRate import getInterfaceRateAndSeq
from utils.readFoldX import readFoldXResult
from utils.generateGraph import generate_residue_graph
from utils.resFeature import getAAOneHotPhys
from utils.getSASA import getDSSP,getRD
from utils.readFoldX import readFoldXResult

if __name__ == '__main__':
    args=get_args()
    print(args)

    complexdict={} # pdbname : [seq1, seq2, bingding_affinity]

    for line in open(args.inputdir+'pdbbind_data.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        complexdict[pdbname]=blocks[1]
            
    exist_files = os.listdir('../graph_res12A/')
    graph_dict=set()
    
    complex_mols = {}
    with open('./data/pdb_mols.txt','r') as f:
        for line in f:
            blocks = re.split('\t|\n',line)
            mols_dict = {}
            for i in range(1,len(blocks)):
                for x in blocks[i]:
                    mols_dict[x] = i-1
            complex_mols[blocks[0]] = mols_dict

    resfeat=getAAOneHotPhys()

    for pdbname in complexdict.keys():
        if pdbname in graph_dict:continue
        pdb_path='/mnt/data/xukeyu/PPA_Pred/PP/'+pdbname+'.ent.pdb'
        logging.info("generate graph:"+pdbname)
        seq,interfaceDict,chainlist,connect=getInterfaceRateAndSeq(pdb_path,complex_mols[pdbname],interfaceDis=args.interfacedis)
        with open('../graph_res12A/'+pdbname+'_seq.picke','wb') as f:
            pickle.dump(seq, f)
        with open('../graph_res12A/'+pdbname+'_interfaceDict.picke','wb') as f:
            pickle.dump(interfaceDict, f)
        with open('../graph_res12A/'+pdbname+'_connect.picke','wb') as f:
            pickle.dump(connect, f)
        energy=readFoldXResult(args.foldxPath,pdbname)
        energy=torch.tensor(energy,dtype=torch.float)
        chainlist=interfaceDict.keys()
        dssp = getDSSP(pdb_path)
        rd = getRD(pdb_path)
        if rd == None or dssp == None:
            continue
        node_feature={}
        logging.info("generate graph feat:"+pdbname)
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
                chain_sign = [0.0]*10
                chain_sign[complex_mols[pdbname][chain]] = 1
                reduise=v.split('_')[1]
                s = seq[pdbname+'_'+chain][1]
                index=int(reduise[1:])-int(s)
                print(index)
                res_key = chain+'_'+str(index+1)
                if res_key not in dssp.keys(): 
                    dssp_feat=[0.0,0.0,0.0,0.0,0.0]
                else:
                    dssp_feat=[dssp[res_key][3],dssp[res_key][7],dssp[res_key][9],dssp[res_key][11],dssp[res_key][13]]#[rSA,...]
                    for j in range(len(dssp_feat)):
                        if dssp_feat[j] == 'NA':
                            dssp_feat[j] = 0.0
                if res_key not in rd.keys():
                    rd_feat=[0.0,0.0]
                else:
                    rd_feat = [rd[res_key][0],rd[res_key][1]]
                    if rd_feat[0] == None:
                        rd_feat[0] = 0.0
                    if rd_feat[1] == None:
                        rd_feat[1] = 0.0
                node_feature[v]=[]
                node_feature[v].append(chain_sign)#chain_sign
                node_feature[v].append(resfeat[reduise[0]])#res feat 
                node_feature[v].append(rd_feat)#res depth 
                node_feature[v].append(dssp_feat)#dssp 
                node_feature[v].append(esm1f_feat[index].tolist())#esm1f 
                node_feature[v].append(esm1v_feat[index].tolist())#esm1v 
                node_feature[v].append(sidechain_center[index])#CaCoor SideChian_CenterCoor 
                node_feature[v].append(sidechain[index])#sidechain angle 
                
        node_features, edge_index,edge_attr=generate_residue_graph(pdbname,node_feature,connect,args.padding)
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
        edge_attr=torch.tensor(edge_attr,dtype=torch.float32)
        torch.save(x.to(torch.device('cpu')),'../graphfeat_8A/'+pdbname+"_x"+'.pth')
        torch.save(edge_index.to(torch.device('cpu')),'../graphfeat_8A/'+pdbname+"_edge_index"+'.pth')
        torch.save(edge_attr.to(torch.device('cpu')),'../graphfeat_8A/'+pdbname+"_edge_attr"+'.pth')
        # torch.save(energy.to(torch.device('cpu')),'./data/skempi/graphfeat/'+pdbname+"_energy"+'.pth')  
    