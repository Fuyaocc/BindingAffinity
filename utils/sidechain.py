import pyrosetta;pyrosetta.init()
from pyrosetta import *
from pyrosetta.teaching import *
init()
import os
import argparse
import pickle
import re

def  sidechain_center(args):
    pdbs=[]

    # exist_pdbs=os.listdir("./sidechain/")
    # filter=set(["1de4","1ogy","1tzn","2wss","3lk4","3sua","3swp","4r8i","6c74"])
    # filter.add('4nzl')
    # filter.add('3l33')
    # filter.add('4dvg')
    # for v in exist_pdbs:
    #     filter.add(v.split('_')[0])

    with open(args.input,'r') as f:
        for line in f:
            pdbs.append(line.split('\t')[0])
    # 定义主链原子的名称
    backbone_atoms = ["N", "CA", "C", "O"]

    
    for pdb in pdbs:
        print(pdb)
        parser = PDBParser()
        structure = parser.get_structure(pdb, './splitByChain/'+pdb+'.txt')  # 替换为实际的 PDB 文件路径

        # 获取第一个模型的第一个链
        model = structure[0]
        chain_name = pdb.split('_')[1]
        chain = model[chain_name]  # 替换为实际的链标识符

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
        with open('./sidechain/'+pdb+'_sidechain_center.picke', 'wb') as file:
            pickle.dump(sidechain, file)
        


def sidechain_angle(args):
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
    description = "you should add those parameter"                   
    parser = argparse.ArgumentParser(description=description)    
    parser.add_argument('--input',type=str,default="./pdb.txt",help='input data dictionary')
    args = parser.parse_args()

    sidechain_angle(args)