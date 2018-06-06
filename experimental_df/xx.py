import pandas as pd

#pdb_list = []
#get      = open('peptides.txt')
#for line in get:
#    line = line.strip('\n')
#    pdb_list.append(line.lower())
#
df             = pd.read_csv            ('experimental_df_connectivity.csv', sep = ',', skipinitialspace = False)
#df             = df.loc[df['PDB'].isin(pdb_list)]
#df             = df.reset_index(drop=True)
#
#df.to_csv('df_connectivity_peptides.csv')
#df             = pd.read_csv            ('df_connectivity_updated.csv', sep = ',', skipinitialspace = False)
#df = df.query('connectivity_index == 1')
print len(df)

pdb_list =df['PDB'].tolist()
pdb_list = (list(set(pdb_list)))
#pdb_list = sorted(pdb_list)
for _ in pdb_list:
    print _

#print len(pdb_list)
#---------------