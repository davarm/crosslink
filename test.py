import pandas as pd

df = pd.read_csv('peptide_df_connectivity.csv')
print df 
pdb_list = df['PDB'].tolist()
pdb_list = list(set(pdb_list))
print len(pdb_list)