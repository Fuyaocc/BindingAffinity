#从结构中获取每条链的所有残基和interface上的残基
#不是看chain之间的interface，而是interact的chain group之间的interface
#顺便从结构中读取序列出来
import Bio
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import logging
import sys
import os
# import pymol
# from pymol import cmd
import math

def addConnect(connect,x,y,dis):
    if(x not in connect.keys()):
        connect[x]=set()
    connect[x].add(y+"="+str(dis))
    # connect[x].add(y+"=1")
    if(y not in connect.keys()):
        connect[y]=set()
    connect[y].add(x+"="+str(dis))
    # connect[x].add(x+"=1")
    return connect

# def getinterfaceWithPymol(pdbPath,threshold=4):
#     cmd.load(pdbPath)
#     cmd.select('proteins', 'polymer.protein')
 
#     # 查找相互作用原子对
#     pairs = cmd.find_pairs("proteins","proteins",cutoff=threshold)

#     # 将原子对转换为蛋白质残基对
#     interfaceRes={}
#     connect=set()
#     for a1, a2 in pairs:
#         at1 = cmd.get_model('%s`%d' % (a1[0], a1[1])).atom[0]
#         at2 = cmd.get_model('%s`%d' % (a2[0], a2[1])).atom[0]

#         if(at1.resn=='UNK'):
#             res1=at1.chain+"_X"+str(at1.resi)
#         else:
#             res1=at1.chain+"_"+Bio.PDB.Polypeptide.three_to_one(at1.resn)+str(at1.resi)
#         if(at2.resn=="UNK"):
#             res2=at2.chain+"_X"+str(at2.resi)
#         else:
#             res2=at2.chain+"_"+Bio.PDB.Polypeptide.three_to_one(at2.resn)+str(at2.resi)

#         if res1 != res2  and res1[0]!=res2[0]:
#             if(res1[0] not in interfaceRes.keys()):
#                 interfaceRes[res1[0]]=set()
#             if(res2[0] not in interfaceRes.keys()):
#                 interfaceRes[res2[0]]=set()
#             interfaceRes[res1[0]].add(res1)
#             interfaceRes[res2[0]].add(res2)
#             connect.add(res1+"_"+res2)
#             connect.add(res2+"_"+res1)
#     cmd.reinitialize()
#     return interfaceRes,connect

def getInterfaceRateAndSeq(pdbPath,mols_dict,interfaceDis=12,mutation=None):
    #pdbName
    pdbName=os.path.basename(os.path.splitext(pdbPath)[0])
    # pdbName=pdbName.split('.')[0].upper()
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("temp", pdbPath)
    interactionInfo=''
    for chain in structure.get_chains():
        interactionInfo=interactionInfo+'_'+chain.get_id()
    interactionInfo=interactionInfo[1:]

    mutation_set = set()
    mutation_idx = []
    for mut in mutation:
        mutation_set.add(mut[1]+'_'+mut[-1]+mut[2:-1])

    #先计算interface residue
    model=structure[0]
    allRes={}  #complex每条链的有效残基
    complexSequence={} #complex中每条链的序列
    CAResName=[]  #残基名称，如E_S38
    CACoor=[] #残基对应的CA坐标
    chainID_in_PDB=set()#无序不重复集
    #提取所有的坐标
    for chain in model:
        minIndex=9999
        maxIndex=-9999 #记录序列的起始位置
        chainID=chain.get_id()
        if chainID==" ":  #有些链是空的
            continue
        allRes[chainID]=set()#空集
        complexSequence[pdbName+'_'+chainID]=list("X"*10240)  #初始化为全为X，序列长为1024的列表
        chainID_in_PDB.add(chainID)
        for res in chain.get_residues():#得到有效残基allRes，序列complexSequence,残基名称及坐标CAResName&CACoor
            resID=res.get_id()[1]
            resName=res.get_resname()
            # print(str(resID)+' '+resName+' '+res.get_id()[0])
            if res.get_id()[0]!=" ":   # 非残基，一般为HOH
                continue
            try:
                if resName == "UNK":#UNK 未知
                    resName = "X"
                else:
                    resName = Bio.PDB.Polypeptide.three_to_one(resName)
            except KeyError:  #不正常的resName
                continue
            try:
                resCoor=res["CA"].get_coord()
            except KeyError:
                continue
            complexSequence[pdbName+'_'+chainID][resID+1000]=resName
            if minIndex>=resID:
                minIndex=resID
            if maxIndex<=resID:
                maxIndex=resID
            allRes[chainID].add(resName+str(resID))
            resCoor=res["CA"].get_coord()
            CAResName.append(chainID+"_"+resName+str(resID))
            if (chainID+"_"+resName+str(resID)) in mutation_set:
                mutation_idx.append(len(CAResName)-1)
            CACoor.append(resCoor)
        complexSequence[pdbName+'_'+chainID]=complexSequence[pdbName+'_'+chainID][minIndex+1000:maxIndex+1000+1]#截取残基链
        complexSequence[pdbName+'_'+chainID]=["".join(complexSequence[pdbName+'_'+chainID]),minIndex] #序列信息和序列起始位置
    #判断PDB中的链和interaction info中的链是否完全一样
    chainID_in_interactionInfo=set(interactionInfo)
    if "_" in chainID_in_interactionInfo:
        chainID_in_interactionInfo.remove("_")
    if not chainID_in_PDB==chainID_in_interactionInfo:
        logging.error("chain in PDB: {}, chain in interaction info {}, not match!".format(str(chainID_in_PDB),str(chainID_in_interactionInfo)))
        #sys.exit()
    #计算distance map
    CACoor=np.array(CACoor)
    dis =  np.linalg.norm(CACoor[:, None, :] - CACoor[None, :, :], axis=-1)
    mask = dis<=interfaceDis
    inside = dis<=5
    resNumber=len(CAResName)
    connect={}
    interfaceRes={}

    interaction_res = set()

    for i in range(resNumber):
        for j in range(i+1,resNumber):
            if CAResName[i].split('_')[1][0] == 'X' or CAResName[j].split('_')[1][0] == 'X':continue
            if mask[i][j] == False or i==j or CAResName[i][0] not in mols_dict.keys() or CAResName[j][0] not in mols_dict.keys():
                continue
            if mols_dict[CAResName[i][0]] != mols_dict[CAResName[j][0]]:
                if CAResName[i][0] not in interfaceRes.keys():
                    interfaceRes[CAResName[i][0]] = set()
                if CAResName[j][0] not in interfaceRes.keys():
                    interfaceRes[CAResName[j][0]] = set()
                interfaceRes[CAResName[j][0]].add(CAResName[j])
                interfaceRes[CAResName[i][0]].add(CAResName[i])
                interaction_res.add(CAResName[j])
                interaction_res.add(CAResName[i])
                connect=addConnect(connect,CAResName[i],CAResName[j],dis[i][j])
    
    for i in range(resNumber):
        for j in range(resNumber):
            if CAResName[i].split('_')[1][0] == 'X' or CAResName[j].split('_')[1][0] == 'X' or CAResName[i][0] not in mols_dict.keys() or CAResName[j][0] not in mols_dict.keys():continue
            if CAResName[i][0] == CAResName[j][0] and i!=j:
                if (math.fabs(int(CAResName[i].split('_')[1][1:])-int(CAResName[j].split('_')[1][1:])) == 1  or inside[i][j]== True) and (CAResName[i] in interaction_res or CAResName[j] in interaction_res):
                    if CAResName[i][0] not in interfaceRes.keys():
                        interfaceRes[CAResName[i][0]] = set()
                    if CAResName[j][0] not in interfaceRes.keys():
                        interfaceRes[CAResName[j][0]] = set()
                    interfaceRes[CAResName[j][0]].add(CAResName[j])
                    interfaceRes[CAResName[i][0]].add(CAResName[i])
                    connect=addConnect(connect,CAResName[i],CAResName[j],-dis[i][j])
    
    for idx in mutation_idx:
        if CAResName[idx][0] in interfaceRes.keys() and CAResName[idx] in interfaceRes[CAResName[idx][0]]:continue
        for j in range(resNumber):
            if j != idx:
                if CAResName[j].split('_')[1][0] == 'X' or CAResName[j][0] not in mols_dict.keys():continue
                if mask[idx][j]:
                    if CAResName[idx][0] not in interfaceRes.keys():
                        interfaceRes[CAResName[idx][0]] = set()
                    if CAResName[j][0] not in interfaceRes.keys():
                        interfaceRes[CAResName[j][0]] = set()
                    interfaceRes[CAResName[idx][0]].add(CAResName[idx])
                    interfaceRes[CAResName[j][0]].add(CAResName[j])
                    connect = addConnect(connect,CAResName[idx],CAResName[j],dis[idx][j])
    return complexSequence,interfaceRes,connect

if __name__ == '__main__':
    seq,interfaceDict,_,connect=getInterfaceRateAndSeq('/mnt/data/xukeyu/data/pdbs/1ay7.pdb','A_B')
    print(seq)
    print(connect)
