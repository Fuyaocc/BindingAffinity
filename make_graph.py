import os
import re
import pickle
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from utils.parse_args import get_args
from utils.getInterfaceRate import getInterfaceRateAndSeq

if __name__ == '__main__':
    args=get_args()
    print(args)

    complexdict={} # pdbname : [seq1, seq2, bingding_affinity]

    for line in open(args.inputdir):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        complexdict[pdbname]=blocks[1]
            
    exist_files = os.listdir('../graph_8A/')
    graph_dict=set()
    for file in exist_files:
        graph_dict.add(file.split("_")[0])
    
    complex_mols = {}
    with open('./data/pdb_mols.txt','r') as f:
        for line in f:
            blocks = re.split('\t|\n',line)
            mols_dict = {}
            for i in range(1,len(blocks)):
                for x in blocks[i]:
                    mols_dict[x] = i-1
            complex_mols[blocks[0]] = mols_dict
    # print(len(complex_mols.keys()))
    for pdbname in complexdict.keys():
        if pdbname in graph_dict:continue
        pdb_path='/mnt/data/xukeyu/data/pdbs/'+pdbname+'.pdb'
        logging.info(pdbname)
        seq,interfaceDict,chainlist,connect=getInterfaceRateAndSeq(pdb_path,complex_mols[pdbname],interfaceDis=args.interfacedis)
        with open('../graph_8A/'+pdbname+'_seq.picke','wb') as f:
            pickle.dump(seq, f)
        with open('../graph_8A/'+pdbname+'_interfaceDict.picke','wb') as f:
            pickle.dump(interfaceDict, f)
        with open('../graph_8A/'+pdbname+'_connect.picke','wb') as f:
            pickle.dump(connect, f)