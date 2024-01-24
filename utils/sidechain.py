import pyrosetta;pyrosetta.init()
from pyrosetta import pose_from_sequence
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
            pdbs.append(re.split('\t|\n',line)[0][:4])
    # 定义主链原子的名称
    backbone_atoms = ["N", "CA", "C", "O"]

    exist_files = os.listdir(out_path)
    for pdb in pdbs:
        print(pdb)
        parser = PDBParser()
        structure = parser.get_structure(pdb, pdbs_path+pdb+'.pdb')  # 替换为实际的 PDB 文件路径

        # 获取第一个模型
        model = structure[0]
        
        for chain in model:
            chain_name = chain.get_id()
            if pdb.lower()+'_'+chain_name+'_sidechain_center.picke' in exist_files:continue
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
            print('sidechain_center : '+pdb.lower()+'_'+chain_name)
            print(len(sidechain[0]))
            with open(out_path+pdb.lower()+'_'+chain_name+'_sidechain_center.picke', 'wb') as file:
                pickle.dump(sidechain, file)
        


def sidechain_angle(dataset_path,seq_path,out_path):
    pdbs=[]

    exist_pdbs=os.listdir(out_path)

    with open(seq_path,'r') as f:
        for line in f:
            v=re.split("\t|\n",line)
            key = v[0][:4].lower()+v[0][-2:]
            pdbs.append(key)

    seqdic={}
    with open(seq_path,'r') as f:
        for line in f:
            v=re.split("\t|\n",line)
            key = v[0][:4].lower()+v[0][-2:]
            seqdic[key]=len(v[1])

    for pdb in pdbs:
        # if pdb[:-2] in filter:continue
        if pdb+'_sidechain.picke' in exist_pdbs:continue
        pose = pose_from_pdb('/mnt/data/xukeyu/PPA_Pred/data/split_by_chain/'+pdb+'.pdb')
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
        with open(out_path+pdb+'_sidechain.picke', 'wb') as file:
            pickle.dump(sidechain, file)
        


if __name__ == '__main__':
    dataset_path = '/mnt/data/xukeyu/PPA_Pred/wt_pdbs.txt'
    pdbs_path = '/mnt/data/xukeyu/PPA_Pred/data/skempi/'
    pdbchains_info_path = '/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/pdb_mols.txt'
    out_path = '/mnt/data/xukeyu/PPA_Pred/feats/sidechain/'
    seq_path = '/mnt/data/xukeyu/PPA_Pred/wt_pdbs_seq.txt'
    chain_group = {}
    for pdbname in dataset_path:
        chains = []
        names = pdbname.split('-')[0].split('_')[1:]
        for name in names:
            for c in name:
                chains.append(c)
        chain_group[pdbname] = chains
    # with open(pdbchains_info_path,'r') as f:
    #     for line in f:
    #         b = re.split('\t|\n',line)
    #         chains = set()
    #         for i in range(1,len(b)):
    #             for c in b[i]:
    #                 chains.add(c)
    #         chain_group[b[0]] = chains
    # sidechain_center(dataset_path,pdbs_path,out_path,chain_group)
    sidechain_angle(dataset_path,seq_path,out_path)