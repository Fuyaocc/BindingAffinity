import pyrosetta;pyrosetta.init()
from pyrosetta import *
from pyrosetta.teaching import *
init()
import os
import pickle
import re
import Bio
from Bio.PDB.PDBParser import PDBParser

def  sidechain_center(dataset_path,pdbs_path,out_path,chain_group):
    pdbs = []
    with open(dataset_path,'r') as f:
        for line in f:
            pdbs.append(line.split('\t')[0])
    # 定义主链原子的名称
    backbone_atoms = ["N", "CA", "C", "O"]

    for pdb in pdbs:
        print(pdb)
        parser = PDBParser()
        structure = parser.get_structure(pdb, pdbs_path+pdb+'.pdb')  # 替换为实际的 PDB 文件路径

        # 获取第一个模型
        model = structure[0]
        
        for chain in model:
            chain_name = chain.get_id()
            sidechain=[]
            for residue in chain:
                try:
                    coords = residue['CA'].get_coord().tolist()
                except:
                    for a in backbone_atoms:
                        try:
                            coords = residue[a].get_coord().tolist()
                            break
                        except: KeyError
                # 获取残基的侧链原子
                sidechain_atoms = [atom for atom in residue.get_atoms() if atom.get_name() not in backbone_atoms]
                # 计算侧链的重心
                if sidechain_atoms:
                    sidechain_center = sum(atom.get_coord() for atom in sidechain_atoms) / len(sidechain_atoms)
                    sidechain_center = sidechain_center.tolist()
                else:
                    sidechain_center = coords
                coords  += sidechain_center
                sidechain.append(coords)
            print('sidechain_center : '+pdb+'_'+chain_name)
            with open(out_path+pdb+'_'+chain_name+'_sidechain_center.picke', 'wb') as file:
                pickle.dump(sidechain, file)
        


def sidechain_angle(args,dataset_path,out_path):
    pdbs=[]

    exist_pdbs=os.listdir("./sidechain/")
    filter=set(["1de4","1ogy","1tzn","2wss","3lk4","3sua","3swp","4r8i","6c74"])
    filter.add('4nzl')
    filter.add('3l33')
    filter.add('4dvg')
    for v in exist_pdbs:
        filter.add(v.split('_')[0])

    with open(args.input,'r') as f:
        for line in f:
            pdbs.append(line.split('\t')[0])

    seqdic={}
    with open("/home/ysbgs/xky/pdb_chain.txt",'r') as f:
        for line in f:
            v=re.split("\t|\n",line)
            seqdic[v[0]]=len(v[1])

    for pdb in pdbs:
        if pdb[:-2] in filter:continue
        
        pose=pose_from_file('./splitByChain/'+pdb+'.txt')
        sidechain=[]
        for i in range(1,seqdic[pdb]+1):
            res_sidechain=[]
            try:
                phi=pose.phi(i)
            except Exception :
                phi=0.0
            res_sidechain.append(phi)
            try:
                psi=pose.psi(i)
            except Exception :
                psi=0.0
            res_sidechain.append(psi)
            for j in range(1,5):
                chi=0.0
                try:
                    chi=pose.chi(j,i)
                except Exception :
                    chi=0.0
                res_sidechain.append(chi)
            sidechain.append(res_sidechain)
        with open('./sidechain/'+pdb+'_sidechain.picke', 'wb') as file:
            pickle.dump(sidechain, file)
        


if __name__ == '__main__':
    dataset_path = '/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/skempi_data.txt'
    pdbs_path = '/mnt/data/xukeyu/PPA_Pred/data/skempi/'
    pdbchains_info_path = '/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/pdb_mols.txt'
    out_path = '/mnt/data/xukeyu/PPA_Pred/feats/sidechain/'
    chain_group = {}
    with open(pdbchains_info_path,'r') as f:
        for line in f:
            b = re.split('\t|\n',line)
            chains = set()
            for i in range(1,len(b)):
                for c in b[i]:
                    chains.add(c)
            chain_group[b[0]] = chains
    sidechain_center(dataset_path,pdbs_path,out_path,chain_group)
    # sidechain_angle(dataset_path,pdbs_path,out_path)