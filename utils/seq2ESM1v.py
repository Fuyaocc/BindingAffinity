#生成esm1v的信息
import esm,torch
import logging
import argparse
import re
import os
def seq2ESM1v(seq, model, alphabet, batch_converter, device):
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
    
    res=torch.Tensor(res).to(device)
    # print(res)
    return res


if __name__ == '__main__':
    description = "you should add those parameter"                   
    parser = argparse.ArgumentParser(description=description)    
    parser.add_argument("--input",default="/mnt/data/xukeyu/PPA_Pred/skempi_seq.txt",help="Output directory, defaults to tmp")
    parser.add_argument('--device',default="cuda:0",help="device")
    args = parser.parse_args()
    logging.info("esm1v model loading")
    esm1v_model_location="/mnt/data/xukeyu/PPA_Pred/feats/esmfeature/esm1v_t33_650M_UR90S_1.pt"
    esm1v_model, alphabet = esm.pretrained.load_model_and_alphabet(esm1v_model_location)
    batch_converter = alphabet.get_batch_converter()
    esm1v_model.to(args.device)
    logging.info("esm1v model load finish")

    pdbs={}
    exist_pdb=set()
    files=os.listdir('/mnt/data/xukeyu/PPA_Pred/esmfeature_skempi/esm1v/')
    for file in files:
        exist_pdb.add(file.split('.')[0])
    with open(args.input,'r') as f:
        for line in f:
            v=re.split('\t|\n',line)
            pdbs[v[0]]=v[1]
    too_long=set()
    for k,v in pdbs.items():
        if k in exist_pdb:continue
        if   len(v)>1024:
            too_long.add(k.split('_')[0])
            continue
        ret=seq2ESM1v(v,esm1v_model,alphabet,batch_converter,args.device)
        print(k)
        torch.save(ret.to(torch.device('cpu')),'/mnt/data/xukeyu/PPA_Pred/esmfeature_skempi/esm1v/'+k+'.pth')
    print(too_long)