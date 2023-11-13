import os
import re
import pickle
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from utils.parse_args import get_args
from utils.resFeature import getAAOneHotPhys
from utils.getInterfaceRate import getInterfaceRateAndSeq


if __name__ == '__main__':
    args=get_args()
    print(args)

    complexdict={} # pdbname : [seq1, seq2, bingding_affinity]

    for line in open(args.inputDir):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        complexdict[pdbname]=blocks[1]
        
    # filter_set=set(["1de4","1ogy","1tzn","2wss","3lk4","3sua","3swp","4r8i","6c74"])#data in test_set or dirty_data
    filter_set=set()
    # for line in open(args.inputDir+'test_set.txt'):
    #     blocks=re.split('\t|\n',line)
    #     filter_set.add(blocks[0])
    
    exist_files=[]
    graph_dict=set()
    # for file in exist_files:
    #     graph_dict.add(file.split(".")[0])
        
    resfeat=getAAOneHotPhys()

    featureList=[]
    labelList=[]
    i=0
    maxlen=0
    for pdbname in complexdict.keys():
        
        if pdbname in filter_set:continue
        if pdbname in graph_dict:continue
        pdb_path='/home/ysbgs/xky/cleanpdb/'+pdbname+'.pdb'
        # energy=readFoldXResult(args.foldxPath,pdbname)
        # energy=torch.tensor(energy,dtype=torch.float)
        # mutation = pdbname.split('-')[1].split('+')
        seq,interfaceDict,chainlist,connect=getInterfaceRateAndSeq(pdb_path,interfaceDis=args.interfacedis)
        logging.info(pdbname)
        with open('./data/graphfeat/'+pdbname+'_seq.picke','wb') as f:
            pickle.dump(seq, f)
        with open('./data/graphfeat/'+pdbname+'_interfaceDict.picke','wb') as f:
            pickle.dump(interfaceDict, f)
        with open('./data/graphfeat/'+pdbname+'_connect.picke','wb') as f:
            pickle.dump(connect, f)
        # with open('./data/graph_4A_12A/graph_info/'+pdbname+'_connect.picke', 'rb') as file:
        #     loaded_data = pickle.load(file)
        # print(loaded_data)
        # dssp=getDSSP(pdb_path)
        # node_feature={}
        # logging.info("generate graph :"+pdbname)
        # for chain in chainlist:
        #     reslist=interfaceDict[chain]
        #     esm1f_feat=torch.load('./data/esmfeature/strute_emb/'+pdbname+'_'+chain+'.pth')
        #     esm1f_feat=F.avg_pool1d(esm1f_feat,16,16)
        #     # le_feat=torch.load('./data/lefeature/'+pdbname+'_'+chain+'.pth')
        #     # le_feat=F.avg_pool1d(le_feat,16,16)
        #     for v in reslist:
        #         reduise=v.split('_')[1]
        #         index=int(reduise[1:])-1
        #         if (chain,index+1) not in dssp.keys(): 
        #             other_feat=[0.0]
        #         else:          
        #             other_feat=[dssp[(chain,index+1)][3]]#[rSA,...]
        #         node_feature[v]=[]
        #         node_feature[v].append(resfeat[reduise[0]])
        #         node_feature[v].append(other_feat)
        #         # node_feature[v].append(le_feat[index].tolist())
        #         node_feature[v].append(esm1f_feat[index].tolist())
                
        # node_features, edge_index,edge_attr=generate_residue_graph(pdbname,node_feature,connect,args.padding)
        # if(len(node_feature)==0):continue
        
        # x = torch.tensor(node_features, dtype=torch.float)
        # edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
        # edge_attr=torch.tensor(edge_attr,dtype=torch.float)
        # torch.save(x.to(torch.device('cpu')),'./data/graph/skempi/'+pdbname+"_x"+'.pth')
        # torch.save(edge_index.to(torch.device('cpu')),'./data/graph/skempi/'+pdbname+"edge_index"+'.pth')
        # torch.save(edge_attr.to(torch.device('cpu')),'./data/graph/skempi/'+pdbname+"_edge_attr"+'.pth')
        # torch.save(energy.to(torch.device('cpu')),'./data/graph/skempi/'+pdbname+"_energy"+'.pth')      
        # data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=complexdict[pdbname],name=pdbname,energy=energy)
