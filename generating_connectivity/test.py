import pandas as pd 
df = pd.read_csv('seq_length.txt', sep = ',',skipinitialspace = False)
# print df

xx = []
for index, row in df.iterrows():
	row = row.tolist()
	if row not in xx:
		xx.append(row)

seq = {}
dis = {}
for _ in xx:
	seq[_[0]] = _[2]
	dis[_[0]] = _[1]

print seq

df2  = pd.read_csv('test_connectivity.csv', sep = ',',skipinitialspace = False)

def seq_length(pdb):
	return (seq[pdb.upper()])
df2['seq_length'] = df['PDB'].apply(seq_length)

def no_disulfides(pdb):
	return (dis[pdb.upper()])
df2['no_disulfides'] = df['PDB'].apply(no_disulfides)

df2.to_csv('test_connectivity.csv', index = 'False')