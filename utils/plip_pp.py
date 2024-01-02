from plip.structure.preparation import PDBComplex

my_mol = PDBComplex()
my_mol.load_pdb('/mnt/data/xukeyu/PPA_Pred/PP/1a2k.ent.pdb') # Load the PDB file into PLIP class
print(my_mol) # Shows name of structure and ligand binding sites
my_mol.analyze()
print(my_mol.interaction_sets)
my_interactions = my_mol.interaction_sets['SO4:E:222'] # Contains all interaction data

# Now print numbers of all residues taking part in pi-stacking
print([pistack.resnr for pistack in my_interactions.pistacking]) # Prints [84, 129]