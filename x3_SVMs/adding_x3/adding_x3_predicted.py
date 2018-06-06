import pandas as pd
import numpy as np
df = pd.read_csv('experimental_df.csv', sep = ',',skipinitialspace = False)
# df   = df.loc[df['x1'] == (-60)]
# df   = df.loc[df['x1b'] == (-60)]
# print df 

# x3_prdicted_df = pd.read_csv('x3_predicted.txt', sep = ' ',skipinitialspace = False)
x3_dict = {}
get = open('x3_predicted.txt','r')
for line in get:
	line = line.strip('\n')
	line = line.replace(' ','')
	# print line 
	line = line.split(',')
	
	

	#if line[2] == '0':
	#	line[2] =90

	#if line[2] == '1':
	#	line[2] = -90

	x3_dict[line[0]+line[1]] = line[2]

# print x3_dict
#----------------------------------
# Add the x3 predicted angles to the df based on PDB and residue
#----------------------------------
# print x3_dict
for index, row in df.iterrows():


	try:
		x3 = x3_dict[str(row['PDB'])+str(row['residue_1'])]
	except KeyError:
		x3 = 0
		# print row['PDB'],row['residue_1']
	try:
		x3b=  x3_dict[str(row['PDB'])+str(row['residue_2'])]
	except KeyError:
		x3b = 0
		# print row['PDB'],row['residue_2']
	#print x3
	#print x3b 

	df.loc[index,'cys1_x3_pred'] = x3
	df.loc[index,'cys2_x3_pred'] = x3b

# print df 
df = df = df.query('cys1_x3_pred != 0')
df = df = df.query('cys2_x3_pred != 0')

df = df.reset_index(drop = True)
# print df 
print len(df)
print df
# df.to_csv('experimental_df_x3_pred.csv')