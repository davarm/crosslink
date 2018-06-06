import pandas as pd 
import numpy as np
import itertools
import os	
#-----------
#To be used for creating hot arrays-------------------------------------------------------------
#Read the disulfide database into a pandas dataframe
#------------------------------------------------------------------------



df = pd.read_csv('experimental_df.csv', sep = ',',skipinitialspace = False)
# Changing all unassigned chemical shifts to 0
df = df.replace([9999],[np.nan])
df = df.replace([9999.999],[np.nan])
df = df.replace([0],[np.nan])

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
df['x3' ]  = df['x3'].apply(x3_rounded)


	
#------------------------------------------------------------------------
# REMOVE ALL Cysteine residues where dihedral angles are outliers
#-----------------------------------------------------1-------------------
configuration=[60,-60,180,90,-90]
df=df[df['x3' ].isin(configuration)]
df=df[df['x1' ].isin(configuration)]
df=df[df['x2' ].isin(configuration)]
df=df[df['x1b'].isin(configuration)]
df=df[df['x2b'].isin(configuration)]


#------------------------------------------------------------------------
# ADDING IN HOT ARRAYS REQUIRED AS INPUTS AND FOR TESTING
#------------------------------------------------------------------------

#X2 Angles
def x2_array(x2):
	configuration                         = [-60,180,60]
	x2_hot_array                          = [0. for _ in range(3)]
	x2_hot_array[configuration.index(x2)] = 1
	x2_hot_array                          = [str(x) for x in x2_hot_array]
	x2_hot_array                          = ",".join(x2_hot_array)
	return (x2_hot_array)

df['x2_array']  = df['x2'].apply(x2_array)
df['x2b_array']  = df['x2b'].apply(x2_array)

#X3 Angles
# def x3_array(x3):
	# if x3 == -90:
		# x3_hot_array = '1'
	# if x3 == 90:
		# x3_hot_array = '0'
	# return (x3_hot_array)

# df['x3_array']  = df['x3'].apply(x3_array)
	
#X1 Angles
def x1_array(x1):
	configuration                         = [-60,180,60]
	x1_hot_array                          = [0. for _ in range(3)]
	x1_hot_array[configuration.index(x1)] = 1
	x1_hot_array                          = [str(x) for x in x1_hot_array]
	x1_hot_array                          = ",".join(x1_hot_array)
	return (x1_hot_array)

df['x1_array']  = df['x1'].apply(x1_array)
df['x1b_array']  = df['x1b'].apply(x1_array)

df['x2_array']  = df['x2'].apply(x2_array)
df['x2b_array']  = df['x2b'].apply(x2_array)

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


#Define classifying amino acid on large,small etc:
charged = ['R','K','H','D','G']
polar   = ['S','T','N','Q']
Cys = ['C','c']
Gly = ['G']
Pro = ['P']
small_hydro = ['A','V','I','L']
large_hydro = ['M','F','Y','W']

def classify_amino(resiude):
	res_status_array = [0. for _ in range(7)]  

	if resiude in charged:
		res_status_array[0] = 1

	if resiude in polar:
		res_status_array[1] = 1

	if resiude in Cys:
		res_status_array[2] = 1

	if resiude in Gly:
		res_status_array[3] = 1

	if resiude in Pro:
		res_status_array[4] = 1

	if resiude in small_hydro:
		res_status_array[5] = 1

	if resiude in large_hydro:
		res_status_array[6] = 1

	res_status_array                                     = [str(x) for x in res_status_array]
	res_status_array                                     = ",".join(res_status_array)
	return(res_status_array)

df['cys1_b_residue_array']  = df['Cys1_before_residue'].apply(classify_amino)
df['cys2_b_residue_array']  = df['Cys2_before_residue'].apply(classify_amino)
df['cys1_a_residue_array']  = df['Cys1_after_residue'].apply(classify_amino)
df['cys2_a_residue_array']  = df['Cys2_after_residue'].apply(classify_amino)
#Amino array

	
	
	
#------------------------------------------------------------------------
# Create df_split, where divide the cysteines into invidiual residues
#Use dictionary to make sure columns are the same
#------------------------------------------------------------------------
#FOR THE FIRST CYS

new_columns=['PDB',
			'x1',
			'x3',
			'x1_array'
			'cys1_phi',
			'cys1_psi',
			'cys1_before_psi',
			'cys1_ss_array',
			'cys1_Ha',
			'cys1_Ca',
			'cys1_Cb',
			'cys1_N',
			'cys1_Hn',
			'cys1_before Ca',
			'cys1_after Ca',
			'cys1_before Ha',
			'cys1_after Ha',
			'cys1_b_residue_array',
			'cys1_a_residue_array',
			'cys1_residue']
	
	
	
df_  = pd.DataFrame(index = None,columns = new_columns)
df_  =  pd.DataFrame({ 
				'PDB' 				: df['PDB'],
				'x1'  				: df['x1'],
				'x3'  				: df['x3'],
				'x1_array'  		: df['x1_array'],
				'cys1_phi'  		: df['phi'],
				'cys1_psi'  		: df['psi'],
				'cys1_before_psi'   : df['Cys1_before_psi'],
				'cys1_ss_array'     : df['cys1_ss_array'],
				'cys1_Ha'           : df['Cys1_HA'],
				'cys1_Ca'           : df['Cys1_CA'],
				'cys1_Cb'           : df['Cys1_CB'],
				'cys1_N'            : df['Cys1_N'],
				'cys1_Hn'           : df['Cys1_HN'],
				'cys1_before Ca'    : df['Cys1_before_CA'],
				'cys1_after Ca'     : df['Cys1_after_CA'],
				'cys1_before Ha'    : df['Cys1_before_HA'],
				'cys1_after Ha'     : df['Cys1_after_HA'],
				'cys1_b_residue_array'            : df['cys1_b_residue_array'],
				'cys1_a_residue_array'            : df['cys1_a_residue_array'],
				'cys1_residue'      : df['residue_1']})	

 		                                  
	
# df_ = df_.astype(str)

df__  = pd.DataFrame(index = None,columns = new_columns)
df__  =  pd.DataFrame({ 
				'PDB' 				: df['PDB'],
				'x1'  				: df['x1b'],
				'x3'  				: df['x3'],
				'x1_array'  		: df['x1b_array'],
				'cys1_phi'  		: df['phi_x'],
				'cys1_psi'  		: df['psi_x'],
				'cys1_before_psi'   : df['Cys2_before_psi'],
				'cys1_ss_array'     : df['cys2_ss_array'],
				'cys1_Ha'           : df['Cys2_HA'],
				'cys1_Ca'           : df['Cys2_CA'],
				'cys1_Cb'           : df['Cys2_CB'],
				'cys1_N'            : df['Cys2_N'],
				'cys1_Hn'           : df['Cys2_HN'],
				'cys1_before Ca'    : df['Cys2_before_CA'],
				'cys1_after Ca'     : df['Cys2_after_CA'],
				'cys1_before Ha'    : df['Cys2_before_HA'],
				'cys1_after Ha'     : df['Cys2_after_HA'],
				'cys1_b_residue_array'            : df['cys2_b_residue_array'],
				'cys1_a_residue_array'            : df['cys2_a_residue_array'],
				'cys1_residue'      : df['residue_2']})	                             
			                    

# df__ = df__.astype(str)

df_split = df_.append(df__, ignore_index = True)
df_split = df_split[['PDB',
					'x1',
					'x3',
					'x1_array',
					'cys1_phi',
					'cys1_psi',
					'cys1_before_psi',
					'cys1_ss_array',
					'cys1_Ha',
					'cys1_Ca',
					'cys1_Cb',
					'cys1_N',
					'cys1_Hn',
					'cys1_before Ca',
					'cys1_after Ca',
					'cys1_before Ha',
					'cys1_after Ha',
					'cys1_b_residue_array',
					'cys1_a_residue_array',
					'cys1_residue']]
	
# df_split = df_split.astype(str)
nuclei_list = ['cys1_Ha',
'cys1_Ca',
'cys1_Cb',
'cys1_N',
'cys1_Hn',
'cys1_before Ca',
'cys1_after Ca',
'cys1_before Ha',
'cys1_after Ha']

print len(df_split)
df_split = df_split.dropna(subset=[nuclei_list], thresh = len(nuclei_list)-2)
df_split = df_split.replace(np.nan,0)
df_split = df_split.loc[df_split['cys1_Cb'] != 0]

print len(df_split)
df_split = df_split.reset_index(drop = True)
# df_split = df_split.fillna(df_split.mean())
df_split.to_csv('Cys_database.csv',index=False)









