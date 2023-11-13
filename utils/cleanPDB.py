import os

if __name__ == '__main__':
    path1='../../protein_ligand/pdbs'
    path2='../../protein_ligand'
    files= os.listdir(path1)
    flag=False
    for file in files:
        if not os.path.isdir(file): 
            infp=path1+'/'+file
            outfp=path2+'/'+file.split('.')[0]+'.pdb'
            with open(infp, "r") as inputFile,open(outfp,"w") as outFile:
                for line in inputFile:
                    if line.startswith('MODEL') and line[12:14]!=' 1':
                        flag=True
                        continue
                    elif line.startswith("ENDMDL"): 
                        flag=False
                        continue
                    if  (line.startswith("ATOM") or line.startswith("SHEET") or line.startswith("TER") ) and flag==False:
                        if line.startswith("ATOM") and line[17:20] == 'UNK':
                            continue
                        if line.startswith("HETATM") and line[17:20] == 'HOH':
                            continue
                        if line.startswith("ATOM"):
                            line = line[:26] + " " + line[27:]
                        outFile.write(line)