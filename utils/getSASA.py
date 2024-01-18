from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import DSSP,ResidueDepth
import os
import sys
import logging

#从pdb生成对应的dssp
def getDSSP(pdbFile):
    if os.path.exists(pdbFile):
        p=PDBParser(QUIET=True)
        structure=p.get_structure("tmp",pdbFile)
        model=structure[0]
        try:
            dssp=DSSP(model,pdbFile,dssp='mkdssp')
        except Exception as e:
            logging.error("can't cal dssp :{}".format(pdbFile))
            logging.error(e)
            return None
        return dssp
    else:
        logging.error("no such pdb:{}".format(pdbFile))
        sys.exit()
    
#从dssp中获取acc
def getAccFromDSSP(dssp,mutation):
    chain=mutation[1]
    wtRes=mutation[0]
    muRes=mutation[-1]
    mutationSite=int(mutation[2:-1])
    try:
        siteDssp=dssp[(chain,mutationSite)]
        if siteDssp[1]!=wtRes:  #mutation中信息和结构中信息不匹配？
            logging.info("res not match in {} and {}".format(str(mutation),str(siteDssp)))
            sys.exit()
    except:
        logging.error("unknown error in dssp")
        sys.exit()
    ss=siteDssp[2]
    acc=float(siteDssp[3])
    return acc

def getRD(pdb_path):
    if os.path.exists(pdb_path):
        p=PDBParser(QUIET=True)
        try:
            structure=p.get_structure("tmp",pdb_path)
            model=structure[0]
            rd = ResidueDepth(model)
        except Exception as e:
            logging.info('can not cal'+pdb_path)
            logging.info(e)
            return None
    else:
        logging.error("no such pdb:{}".format(pdb_path))
        sys.exit()
    start = 1
    chain_k = '*'
    rerd={}
    for k in rd.keys():
        if k[0]!=chain_k:
            chain_k=k[0]
            start=1
        rerd[k[0]+'_'+str(start)] = rd[k]
        start+=1
    return rerd