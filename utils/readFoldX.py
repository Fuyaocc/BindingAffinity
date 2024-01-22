import re
import os

def readFoldXResult(path,pdbname):
    foldxres=[0.]*25
    result_path=path+"Interaction_"+pdbname+".ent_AC.fxout"
    if os.path.exists(result_path)==False:
        os.system("cd /mnt/data/xukeyu/PPA_Pred/data/skempi/"+"&&"+f'/mnt/data/xukeyu/PPA_Pred/foldx/foldx_20241231 --command=AnalyseComplex --pdb="{pdbname}.pdb" --complexWithDNA=false  --output-dir="/mnt/data/xukeyu/PPA_Pred/foldx/foldx_result/"')
    with open(result_path,"r") as f:
        for line in f:
            index=0
            if line.startswith("./"):
                energy=re.split("\t|\n",line)
                for i in range(len(energy)):
                    if i<6 or i==27 or i==32 :continue
                    foldxres[index]+=float(energy[i])
                    index+=1
    return foldxres
