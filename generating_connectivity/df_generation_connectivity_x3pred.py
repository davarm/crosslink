import pandas as pd 
import numpy as np
import itertools
import os	
															
#-----------
#To be used for creating hot arrays-------------------------------------------------------------
#Read the disulfide database into a pandas dataframe
#------------------------------------------------------------------------


df = pd.read_csv('experimental_df_x3pred_refined.csv', sep = ',',skipinitialspace = False)
# Changing all unassigned chemical shifts to 0
df = df.replace([9999.000],['0'])
df = df.replace([9999],['0'])

# print df
#df = df.loc[df['Cys1_after_residue'] != ('TERMINAL')]
#df = df.loc[df['Cys2_before_residue'] != ('TERMINAL')]
#df = df.loc[df['Cys2_before_residue'] != ('X')]
#df = df.loc[df['Cys1_after_residue'] != ('X')]
#
#df = df.loc[df['Cys1_before_residue'] != ('X')]
#df = df.loc[df['Cys2_after_residue'] != ('X')]
df = df.replace([9999],['0'])
print 'XXXXXXX',len(df)
# print df['PDB']
#------------------------------------
#Generate a list of PDB numbers
#------------------------------------
pdb_list       =  df['PDB'].tolist()
pdb_list       =  list(set(pdb_list))

#------------------------------------------------------------------------
#Applying rounding functions to the cysteine dihedral angles
#------------------------------------------------------------------------
def x1_rounded(x1):
		x1=float(x1)
		if (x1 <=  90) & (x1 >= 30):
			x1= 60
		if (x1 >= -90)  & (x1 <= -30):
			x1 = -60   
		if (x1 <=  180) & (x1 >= 150):
			x1= 180
		if (x1 >= -180) & (x1 <= -150):
			x1=180
		return(x1)

def x2_rounded(x2):
		x2=float(x2)
		if (x2 <=  120) & (x2 >= 30):
			x2= 60
		if (x2 >= -120) & (x2 <= -30):
			x2 = -60   
		if (x2 <=  180) & (x2 >= 150):
			x2= 180
		if (x2 >= -180) & (x2 <= -150):
			x2=180
		return(x2)

def x3_rounded(x3):
		x3=float(x3)
		if (x3  <=  120)  & (x3 >= 60):
			x3   = 90
		if (x3  >=  -120) & (x3 <= -60):
			x3   = -90
		return(x3)

#Apply funcitons
df['x1' ]  = df['x1'].apply(x1_rounded)
df['x1b'] = df['x1b'].apply(x1_rounded)
df['x2' ]  = df['x2'].apply(x2_rounded)
df['x2b'] = df['x2b'].apply(x2_rounded)
# df['x3' ]  = df['x3'].apply(x3_rounded)

#------------------------------------------------------------------------
# REMOVE ALL Cysteine residues where dihedral angles are outliers
#-----------------------------------------------------1-------------------
configuration=[60,-60,180]
x3configuration = [-90,90]
# df=df[df['x3' ].isin(x3configuration)]

df=df[df['x1' ].isin(configuration)]
df=df[df['x2' ].isin(configuration)]
df=df[df['x1b'].isin(configuration)]
df=df[df['x2b'].isin(configuration)]
# print pdb_list


#------------------------------------------------------------------------
# ADDING IN HOT ARRAYS REQUIRED AS INPUTS AND FOR TESTING
#------------------------------------------------------------------------

#X2 Angles
#def x2_array(x2):
#	configuration                         = [60,-60,180]
#	x2_hot_array                          = [0. for _ in range(3)]
#	x2_hot_array[configuration.index(x2)] = 1
#	x2_hot_array                          = [str(x) for x in x2_hot_array]
#	x2_hot_array                          = ",".join(x2_hot_array)
#	return (x2_hot_array)

# df['x2_array']  = df['x2'].apply(x2_array)
# df['x2b_array']  = df['x2b'].apply(x2_array)
#X3 Angles
def x3_array(x3):
	configuration                         = [90,-90]
	x3_hot_array                          = [0. for _ in range(2)]
	x3_hot_array[configuration.index(x3)]= 1
	x3_hot_array                          = [str(x) for x in x3_hot_array]
	x3_hot_array                          = ",".join(x3_hot_array)
	return (x3_hot_array)

df['cys1_x3_array']  = df['cys1_x3_pred'].apply(x3_array)
df['cys2_x3_array']  = df['cys2_x3_pred'].apply(x3_array)

#X1 Angles
def x1_array(x1):
	configuration                         = [60,-60,180]
	x1_hot_array                          = [0. for _ in range(3)]
	x1_hot_array[configuration.index(x1)] = 1
	x1_hot_array                          = [str(x) for x in x1_hot_array]
	x1_hot_array                          = ",".join(x1_hot_array)
	return (x1_hot_array)

df['x1_array']  = df['x1'].apply(x1_array)
df['x1b_array']  = df['x1b'].apply(x1_array)

#Secondary Structure
def changing_ss(ss):
	configuration = ['H','E','L']
	if ss == 'G':
		ss = 'H'
	if ss == 'B':
		ss = 'E'

	if ss == 'I' or ss == 'S' or ss == 'T' or 'ss' == 'C':
		ss = 'L'

	if ss not in configuration:
		ss = 'L'

	return(ss)

def ss_array(ss):
	
	secondary_structure_list = [
		'H'		,
		'E'	,
		'L'] 
	#print ss
	ss_hot_array 									 = [0. for _ in range(3)]  
	ss_hot_array[secondary_structure_list.index(ss)] = 1
	ss_hot_array                                     = [str(x) for x in ss_hot_array]
	ss_hot_array                                     = ",".join(ss_hot_array)
	return (ss_hot_array)

df['Cys1 dssp']  = df['Cys1_SS'].apply(changing_ss)
df['Cys2 dssp']  = df['Cys2_SS'].apply(changing_ss)

df['cys1_ss_array']  = df['Cys1 dssp'].apply(ss_array)
df['cys2_ss_array']  = df['Cys2 dssp'].apply(ss_array)

#Amino array
def amino_array(aa):
	amino_list = [
	'A',                                                                                                                              
	'R',                                                                                                                              
	'N',                                                                                                                              
	'D',                                                                                                                              
	'c',
	'C',                                                                                                                              
	'Q',                                                                                                                              
	'E',                                                                                                                              
	'G',                                                                                                                              
	'H',                                                                                                                              
	'I',                                                                                                                              
	'L',                                                                                                                              
	'K',                                                                                                                              
	'M',                                                                                                                              
	'F',                                                                                                                              
	'P',                                                                                                                              
	'S',                                                                                                                              
	'T',                                                                                                                              
	'W',                                                                                                                              
	'Y',                                                                                                                              
	'V',
	'TERMINAL',
	'X']

	amino_hot_array                       = [0. for _ in range(23)]  
	amino_hot_array[amino_list.index(aa)] = 1
	amino_hot_array                       = [str(x) for x in amino_hot_array]
	amino_hot_array                       = ",".join(amino_hot_array)
	return (amino_hot_array)

df['cys1_b_residue_array']  = df['Cys1_before_residue'].apply(amino_array)
df['cys2_b_residue_array']  = df['Cys2_before_residue'].apply(amino_array)
df['cys1_a_residue_array']  = df['Cys1_after_residue'].apply(amino_array)
df['cys2_a_residue_array']  = df['Cys2_after_residue'].apply(amino_array)


#------------------------------------------------------------------------
# Create df_split, where divide the cysteines into invidiual residues
#Use dictionary to make sure columns are the same
#------------------------------------------------------------------------
#FOR THE FIRST CYS

new_columns=['PDB',
			 'residue_number',
			 'seq_lenght',
			 'no_disulfides',
			 'x1',
			 'phi',
			 'psi',
			 'before psi',
			 'before residue',
			 'after residue',
			 'b_residue_array',
			 'a_residue_array',
			 'ss_array',
			 'x3',
			 'Cb',
			 'Ca',
			 'before Ca',
			 'after Ca',
			 'x3_array',
			 'x1_array',
			 'x1',
			 'Hn',
			 'Ha']	


df_  = pd.DataFrame(index = None,columns = new_columns)
df_  =  pd.DataFrame({                                                
		"PDB"				:df['PDB'],                        
		"residue_number"	:df["residue_1"],                        
		"seq_length"  		:df["seq_length"],                            
		"no_disulfides"  		:df["no_disulfides"],                            
		"x1_array"  		:df["x1_array"],                            
		"phi" 				:df["phi"],                          
		"psi" 				:df["psi"],                          
		"before psi" 		:df["Cys1_before_psi"],                          
		"before residue" 	:df["Cys1_before_residue"],                          
		"after residue" 	:df["Cys1_after_residue"],                          
		"b_residue_array" 	:df["cys1_b_residue_array"],                          
		"a_residue_array"	:df["cys1_a_residue_array"],                                    
		"x3_array"			:df["cys1_x3_array"],                          
		"x3"				:df["cys1_x3_pred"],                                      
		"x1"				:df["x1"],                                      
		"Cb"				:df["Cys1_CB"],                                 
		"Ca"				:df["Cys1_CA"],                                 
		"Ha"				:df["Cys1_HA"],                                 
		"Hn"				:df["Cys1_HN"],                                 
		"N"					:df["Cys1_N"],                                 
		"before Ca"			:df["Cys1_before_CA"],                                 
		"after Ca"			:df["Cys1_after_CA"],
		"before Ha"			:df["Cys1_before_HA"],                                 
		"after Ha"			:df["Cys1_after_HA"],
		"before Hn"			:df["Cys1_before_HN"],                                 
		"after Hn"			:df["Cys1_after_HN"],
		"before Cb"			:df["Cys1_before_CB"],                                 
		"after Cb"			:df["Cys1_after_CB"],                                     
		"ss_array"			:df["cys1_ss_array"]})		 		                                  



df__  = pd.DataFrame(index = None,columns = new_columns)
df__  =  pd.DataFrame({                                                
		"PDB"				:df['PDB'],                        
		"residue_number"	:df["residue_2"],                        
		"seq_length"  		:df["seq_length"],                            
		"no_disulfides"  	:df["no_disulfides"], 
		"x1_array"  		:df["x1b_array"],
		"phi" 				:df["phi_x"],                          
		"psi" 				:df["psi_x"],
		"before psi" 		:df["Cys2_before_psi"],
		"before residue" 	:df["Cys2_before_residue"],                          
		"after residue" 	:df["Cys2_after_residue"],                                 
		"b_residue_array" 	:df["cys2_b_residue_array"],                          
		"a_residue_array"	:df["cys2_a_residue_array"],
		"x3_array"			:df["cys2_x3_array"],
		"x3"			    :df["cys2_x3_pred"],
		"x1"			    :df["x1b"],
		"Cb"				:df["Cys2_CB"],                                 
		"Ca"				:df["Cys2_CA"],                                 
		"Ha"				:df["Cys2_HA"],			
		"Hn"				:df["Cys2_HN"],			
		"N"					:df["Cys2_N"],			
		"before Ca"			:df["Cys2_before_CA"],                                 
		"after Ca"			:df["Cys2_after_CA"],
		"before Ha"			:df["Cys2_before_HA"],                                 
		"after Ha"			:df["Cys2_after_HA"],
		"before Hn"			:df["Cys2_before_HN"],                                 
		"after Hn"			:df["Cys2_after_HN"],
		"before Cb"			:df["Cys2_before_CB"],                                 
		"after Cb"			:df["Cys2_after_CB"],                                  
		"ss_array"			:df["cys2_ss_array"]})	

df_split = df_.append(df__, ignore_index = True)

print len(df_)
# ONly if x3 are the same)





final_database= pd.DataFrame([])#,columns = final_columns)
for pdb in pdb_list:
	cys_list=[]
	cys_list=(df_split.loc[df_split['PDB'] == pdb, 'residue_number']).tolist()
	cys_list=sorted(cys_list, key=lambda x: float(x),reverse=True)
	possible_connections=[]
	possible_connections=list(itertools.combinations(cys_list, 2))
	
	
	
	for possible in possible_connections:
		possible    = list(possible)
		cys1,cys2   = possible[0], possible[1]
		
		cys1_shifts = df_split.loc[(df_split['PDB'] == pdb) & (df_split['residue_number'] == cys1)]
		cys2_shifts = df_split.loc[(df_split['PDB'] == pdb) & (df_split['residue_number'] == cys2)]
	
		cys1_shifts = cys1_shifts.reset_index(drop=True)
		cys2_shifts = cys2_shifts.reset_index(drop=True)
		
		joined		= pd.concat(dict(cys1_shifts = cys1_shifts, cys2_shifts = cys2_shifts),axis=1)
		#------------------------------------------------------------------------
		# IF PDB, CYS1 AND CYS2 ARE IN THE ORIGINAL DF, MEANS THEY ARE A TRUE CONNECTION
		#------------------------------------------------------------------------

		search_result = df.loc[(df['PDB'] == pdb) & (df['residue_1'] == cys1) & (df['residue_2'] == cys2)]
		
		if len(search_result ) == 1:
			joined['connectivity'] = 'true'
			joined['connectivity_array'] = '0,1'
			joined['connectivity_index'] = '1'
		
		if len(search_result ) == 0:
			joined['connectivity'] = 'false'
			joined['connectivity_array'] = '1,0'
			joined['connectivity_index'] = '0'
		if len(search_result ) != 0 and len(search_result) != 1:
			continue 
		
		final_database=final_database.append(pd.DataFrame(joined,index=[0]), ignore_index=True)

#CHANGING COLUMN NAMES
new_names=[]
xx=list(final_database.columns.values)
for _ in xx:
	cys_number= _[0]

	if cys_number == 'cys1_shifts' or cys_number=='cys2_shifts':
		new_names.append(''.join([str(cys_number[0:5]), str(_[1])]))
	else:
		new_names.append(str(_[0]))



#new_column_names=[]
#for i,_ in enumerate(originals):
#	x=str(test[i])
#	new = str(_[0:5])
#	new_column_names.append(''.join([new, x]))

##final_database = final_database.drop(final_database.iloc[0])
final_database.columns = new_names
print final_database
final_database = final_database.loc[(final_database['cys1_x3']) == (final_database['cys2_x3'])]
final_database = final_database.reset_index(drop = True)
final_database['cys_diff'] =  final_database['cys1_residue_number'] - final_database['cys2_residue_number']
final_database = final_database.query('cys_diff != 0')
final_database = final_database.reset_index(drop = True)


final_database.to_csv('experimental_df_connectivity_x3_pred.csv',index=False)	
	