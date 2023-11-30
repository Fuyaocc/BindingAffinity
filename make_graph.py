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

    for line in open(args.inputDir):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        complexdict[pdbname]=blocks[1]
            
    exist_files = os.listdir('../graph/')
    graph_dict=set()
    for file in exist_files:
        graph_dict.add(file.split("_")[0])

    for pdbname in complexdict.keys():
        if pdbname in graph_dict:continue
        pdb_path='/mnt/data/xukeyu/data/pdbs/'+pdbname+'.pdb'
        logging.info(pdbname)
        seq,interfaceDict,chainlist,connect=getInterfaceRateAndSeq(pdb_path,interfaceDis=args.interfacedis)
        with open('../graph/'+pdbname+'_seq.picke','wb') as f:
            pickle.dump(seq, f)
        with open('../graph/'+pdbname+'_interfaceDict.picke','wb') as f:
            pickle.dump(interfaceDict, f)
        with open('../graph/'+pdbname+'_connect.picke','wb') as f:
            pickle.dump(connect, f)