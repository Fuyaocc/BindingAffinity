#生成esm1v的信息
import re
import os
import gc
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import esm,torch
import logging
from Bio.PDB.PDBParser import PDBParser
import esm.inverse_folding

def run_esm1v(esm1v_model_location,seqdata_path,out_path,device):
    logging.info("esm1v_model loading")
    esm1v_model_location = esm1v_model_location
    esm1v_model, alphabet = esm.pretrained.load_model_and_alphabet(esm1v_model_location)
    batch_converter = alphabet.get_batch_converter()
    esm1v_model.to(device)
    for param in esm1v_model.parameters():
        param.requires_grad = False
    esm1v_model = esm1v_model.eval()
    logging.info("esm1v esmif1_model load finish")
    
    exist_pdb=set()
    files=os.listdir(out_path)
    for file in files:
        exist_pdb.add(file.split('.')[0])
    
    pdb_seqs={}
    with open(seqdata_path,'r') as f:
        for line in f:
            v=re.split('\n|\t',line)
            print(v)
            key = v[0]
            for i in range(1,len(v)):
                pdb_seqs[key+'_'+chr(ord('A')+i-1)]=v[i]
    
    for k,v in pdb_seqs.items():
        
        if k in exist_pdb:continue
        
        
        if  len(v)>1024:
            logging.warning(k+' seq is longger than 1024.')
            first = v[:1000]
            second = v[1000:]
            ss = [first,second]
        else:
            ss = [v]
        # get esm1v embedding
        try:
            res=[]
            for s in ss:
                data = [("tmp", s),]
                _, _, batch_tokens = batch_converter(data)
                for i in range(len(s)):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0][i+1]=alphabet.mask_idx  #mask the residue,32
                    with torch.no_grad():
                        x=esm1v_model(batch_tokens_masked.to(device))
                    res.append(x[0][i+1].tolist())
            res=torch.tensor(res)
            logging.info('Finish esm1v\'s embedding : '+k)
            torch.save(res,out_path+k+'.pth')
        except Exception:
            print("pdb imformation not clear :"+k)
    
def run_esmif1(esmif1_model_location,dataset_path,pdbs_path,pdbchains_info_path,out_path,device):
    esmif1_model_location = esmif1_model_location
    esmif1_model, alphabet = esm.pretrained.load_model_and_alphabet(esmif1_model_location)
    for param in esmif1_model.parameters():
        param.requires_grad = False
    esmif1_model = esmif1_model.eval()
    esmif1_model.to(device)
    
    dataset=[]
    with open(dataset_path,'r') as f:
        for line in f:
            pdb=re.split('\t|\n',line)
            dataset.append(pdb[0])
        
    exist_pdb = set()
    for file in os.listdir(out_path):
        exist_pdb.add(file.split('_')[0])
    
    chainGroup = {}
    with open(pdbchains_info_path,'r') as f:
        for line in f:
            b = re.split('\t|\n',line)
            pdbname = b[0].split('_')[0]
            chain = b[0].split('_')[1]
            if pdbname not in chainGroup.keys():
                chainGroup[pdbname] = [chain]
            else:
                chainGroup[pdbname].append(chain)
    
    for pdbname in dataset: #遍历文件夹
        # if pdbname in exist_pdb:continue
        fp=pdbs_path+pdbname+'/ranked_0.pdb'
        parser = PDBParser()
        structure = parser.get_structure("temp", fp)[0]
        try:
            structure = esm.inverse_folding.util.load_structure(fp, chainGroup[pdbname])
            coords, _ = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
            for chain_id in chainGroup[pdbname]:
                if pdbname+'_'+chain_id in exist_pdb: continue
                rep = esm.inverse_folding.multichain_util.get_encoder_output_for_complex(esmif1_model, alphabet, coords, chain_id, device)
                logging.info('Finish esmif1\'s embedding : '+pdbname+'_'+chain_id)
                torch.save(rep.to(torch.device('cpu')),out_path+pdbname+'_'+chain_id+'.pth')
                del rep 
                gc.collect()
        except Exception:
            print('pdb imformation not clear'+pdbname)
    


if __name__ == '__main__':
    device = 'cuda:3'
    dataset_path = '/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/dg_data/Antigen-Antibody.txt'
    seqdata_path = '/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/dg_data/Antigen-Antibody.txt'
    pdbs_path = '/mnt/data/xukeyu/PPA_Pred/nanobody_list_0202_wo_pkl/msa/'
    esm1v_out_path = '/mnt/data/xukeyu/PPA_Pred/feats/esm/pdbbind/esm1v/'
    esmif1_out_path = '/mnt/data/xukeyu/PPA_Pred/feats/esm/pdbbind/esmif1/'
    esm1v_model_location = '/mnt/data/xukeyu/PPA_Pred/feats/esm/esm1v_t33_650M_UR90S_1.pt'
    esmif1_model_location = '/mnt/data/xukeyu/PPA_Pred/feats/esm/esm_if1_gvp4_t16_142M_UR50.pt'
    pdbchains_info_path = '/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/dg_data/antibody_antigen_seq.txt'
    run_esmif1(esmif1_model_location,dataset_path,pdbs_path,pdbchains_info_path,esmif1_out_path,device)
    # run_esm1v(esm1v_model_location,seqdata_path,esm1v_out_path,device)