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

    complexdict={}

    for line in open('/mnt/data/xukeyu/PPA_Pred/foldx/skempi.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        complexdict[pdbname]=0.0
            
    exist_files = os.listdir('../graphfeat_res12A/')
    graph_dict=set()
    for file in exist_files:
        graph_dict.add(file.split('.')[0][:-2])
    
    complex_mols = {}
    with open('./data/pdb_mols.txt','r') as f:
        for line in f:
            blocks = re.split('\t|\n',line)
            mols_dict = {}
            for i in range(1,len(blocks)):
                for x in blocks[i]:
                    mols_dict[x] = i-1
            complex_mols[blocks[0]] = mols_dict
    
    for name in complexdict.keys():
        if len(name) == 4:continue
        tmp = name.split('-')[0]
        chains = tmp.split('_')[1:]
        mols_dict = {}
        for i in range(len(chains)):
            for c in chains[i]:
                mols_dict[c] = i
        complex_mols[name] = mols_dict
    
    complex_mols['3cph']={'A':0,'B':1}
    complex_mols['4cpa']={'A':0,'B':1}
    
    resfeat = getAAOneHotPhys()

    graph_path = '../graph_res12A/'
    feat_path = '../graphfeat_res12A/'

    for pdbname in complexdict.keys():
        # if pdbname in graph_dict:continue
        # pdb_path = '/mnt/data/xukeyu/PPA_Pred/PP/'+pdbname+'.ent.pdb'
        # if pdbname != '3SE3_B_A-YB43M+NB44D+SB47L':continue
        pdb_path = '/mnt/data/xukeyu/PPA_Pred/data/skempi/'+pdbname+'.pdb'
        seq_path = graph_path+pdbname+'_seq.picke'
        interfaceDict_path = graph_path+pdbname+'_interfaceDict.picke'
        connet_path = graph_path+pdbname+'_connect.picke'
        # if os.path.exists(seq_path) == False or os.path.exists(interfaceDict_path) == False or os.path.exists(connet_path) == False:
        if True:
            logging.info("generate graph:"+pdbname)
            if len(pdbname) == 4:
                mutation = None
            else:
                tmp = pdbname.split('-')[1]
                mutation = tmp.split('+')
            seq,interfaceDict,connect=getInterfaceRateAndSeq(pdb_path,complex_mols[pdbname],interfaceDis=args.interfacedis,mutation=mutation)
            with open(seq_path,'wb') as f:
                pickle.dump(seq, f)
            with open(interfaceDict_path,'wb') as f:
                pickle.dump(interfaceDict, f)
            with open(connet_path,'wb') as f:
                pickle.dump(connect, f)
        else:
            logging.info("load graph:"+pdbname)
            with open(seq_path, 'rb') as file:
                seq = pickle.load(file)
            with open(interfaceDict_path, 'rb') as file:
                interfaceDict = pickle.load(file)
            with open(connet_path, 'rb') as file:
                connect = pickle.load(file)
        chainlist=interfaceDict.keys()
        # print(seq)
        # print(interfaceDict)
        # print(complex_mols[pdbname])
        if len(chainlist) > 10:continue
        dssp_path = '../feats/dssp/'+pdbname+'.pickle'
        rd_path = '../feats/rd/'+pdbname+'_rd.pickle'
        if os.path.exists(dssp_path):
            with open(dssp_path, 'rb') as file:
                dssp = pickle.load(file)
        else:
            dssp = getDSSP(pdb_path)
            if dssp != None:
                with open(dssp_path,'wb') as f:
                    pickle.dump(dssp, f)

        if os.path.exists(rd_path):
            with open(rd_path, 'rb') as file:
                rd = pickle.load(file)
        else:
            rd = getRD(pdb_path)
            if rd!= None:
                with open(rd_path,'wb') as f:
                    pickle.dump(rd, f)

        node_feature={}
        logging.info("generate graph feat:"+pdbname)
        # print(chainlist)
        for chain in chainlist:
            seq_chain = seq[pdbname+'_'+chain][0]
            idx_map = {}
            ss = 0
            for o in range(len(seq_chain)):
                if seq_chain[o] != 'X':
                    idx_map[o] = ss
                    ss += 1
            with open('../feats/sidechain/'+pdbname+'_'+chain+'_sidechain.picke', 'rb') as file:
                sidechain = pickle.load(file)
            with open('../feats/sidechain/'+pdbname+'_'+chain+'_sidechain_center.picke', 'rb') as file:
                sidechain_center = pickle.load(file)
            reslist=interfaceDict[chain]
            esm1f_feat=torch.load('../feats/esm/skempi/esmif1/'+pdbname+'_'+chain+'.pth')
            esm1f_feat=F.avg_pool1d(esm1f_feat,16,16)
            esm1v_feat=torch.load('../feats/esm/skempi/esm1v/'+pdbname+'_'+chain+'.pth')
            esm1v_feat=F.avg_pool1d(esm1v_feat,40,40)
            s = seq[pdbname+'_'+chain][1]
            for v in reslist:
                chain_sign = [0.0]*10
                chain_sign[complex_mols[pdbname][chain]] = 1
                reduise=v.split('_')[1]
                index=int(reduise[1:])-int(s)
                res_key = chain+'_'+str(index+1)
                if dssp is None or res_key not in dssp.keys(): 
                    dssp_feat=[0.0,0.0,0.0,0.0,0.0]
                else:
                    dssp_feat=[dssp[res_key][3],dssp[res_key][7],dssp[res_key][9],dssp[res_key][11],dssp[res_key][13]]#[rSA,...]
                    for j in range(len(dssp_feat)):
                        if dssp_feat[j] == 'NA':
                            dssp_feat[j] = 0.0
                if rd is None or res_key not in rd.keys():
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
                node_feature[v].append(esm1f_feat[idx_map[index]].tolist())#esm1f 
                node_feature[v].append(esm1v_feat[idx_map[index]].tolist())#esm1v 
                node_feature[v].append(sidechain[idx_map[index]])#sidechain angle 
                node_feature[v].append(sidechain_center[idx_map[index]])#CaCoor SideChian_CenterCoor 
                
        node_features, edge_index,edge_attr=generate_residue_graph(pdbname,node_feature,connect,args.padding)
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
        edge_attr=torch.tensor(edge_attr,dtype=torch.float32)
        torch.save(x.to(torch.device('cpu')),feat_path+pdbname+"_x"+'.pth')
        torch.save(edge_index.to(torch.device('cpu')),feat_path+pdbname+"_edge_index"+'.pth')
        torch.save(edge_attr.to(torch.device('cpu')),feat_path+pdbname+"_edge_attr"+'.pth')    