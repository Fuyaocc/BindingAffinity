import esm
import numpy as np
import torch
import Bio
from Bio.PDB.PDBParser import PDBParser
import os
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import esm.inverse_folding
from parse_args import get_args
import shutil
import gc
import re

def runesm1v(seq, model, alphabet, batch_converter, device):
    res=[]
    data = [("tmp", seq),]
    _, _, batch_tokens = batch_converter(data)
    for i in range(len(seq)):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0][i+1]=alphabet.mask_idx  #mask the residue,32
        # print(batch_tokens_masked)
        with torch.no_grad():
            x=model(batch_tokens_masked.to(device))
        res.append(x[0][i+1].tolist())
    return torch.Tensor(res).to(device)

# dstpath 目的地址
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        shutil.move(srcfile,dstfile)          #移动文件
        print("move %s -> %s"%( srcfile,dstfile))

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        shutil.copyfile(srcfile,dstfile)      #复制文件
    

if __name__ == '__main__':
    args=get_args()
    exist_files=[]
    esm_dict=set()
    for line in exist_files:
        esm_dict.add(line.split(".")[0][:-2])
    esmif1_model_location='/mnt/data/xukeyu/PPA_Pred/feats/esm/esm_if1_gvp4_t16_142M_UR50.pt'
    model, alphabet = esm.pretrained.load_model_and_alphabet(esmif1_model_location)
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()
    
    files=[]
    with open('/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/skempi_data.txt','r') as f:
        for line in f:
            pdb=re.split('\t|\n',line)
            files.append(pdb[0])
    
    path='/mnt/data/xukeyu/PPA_Pred/data/skempi/'
    too_long=["1ogy","1tzn","2wss","3lk4","3sua","3swp"]
    # f = open('/mnt/data/xukeyu/PPA_Pred/skempi_seq.txt','w')
    
    files=os.listdir('/mnt/data/xukeyu/PPA_Pred/esmfeature_skempi/esm1v/')
    exist_pdb = set()
    for file in files:
        exist_pdb.add(file.split('.')[0])
    
    for file in files: #遍历文件夹
        fp=path+file+'.pdb'
        parser = PDBParser()
        structure = parser.get_structure("temp", fp)
        chainGroup=set()
        logging.info(file)
        for chain in structure.get_chains():
            chainGroup.add(chain.get_id())
        print(chainGroup)
        structure = esm.inverse_folding.util.load_structure(fp, list(chainGroup))
        coords, seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
        print(seqs)
        for chain_id in chainGroup:
            if file.split('.')[0]+'_'+chain_id in exist_pdb: continue
            rep = esm.inverse_folding.multichain_util.get_encoder_output_for_complex(model, alphabet, coords, chain_id)
            print(rep.shape)
            torch.save(rep.to(torch.device('cpu')),'/mnt/data/xukeyu/PPA_Pred/esm/skempi/esmif1/'+file.split('.')[0]+'_'+chain_id+'.pth')
            del rep 
            gc.collect()        
