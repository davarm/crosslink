import pandas as pd 
import numpy as np
import itertools
import os	
															
#-----------
#To be used for creating hot arrays-------------------------------------------------------------
#Read the disulfide database into a pandas dataframe
#------------------------------------------------------------------------

peptide_list = ['G117','2lye','2mfa','2n8e','2n9t','geodia','2mfa']
for peptide in peptide_list:
	df = pd.read_csv(peptide+'_inputs.csv', sep = ',',skipinitialspace = False)
	# Changing all unassigned chemical shifts to 0
	df = df.replace([9999.000],['0'])
	df = df.replace([9999],['0'])
	
	#df = df.loc[df['Cys1_after_residue'] != ('TERMINAL')]
	#df = df.loc[df['Cys2_before_residue'] != ('TERMINAL')]
	#df = df.loc[df['Cys2_before_residue'] != ('X')]
	#df = df.loc[df['Cys1_after_residue'] != ('X')]
	#
	#df = df.loc[df['Cys1_before_residue'] != ('X')]
	#df = df.loc[df['Cys2_after_residue'] != ('X')]
	df = df.replace([9999],['0'])
	print df['PDB']
	
	
	
	
	#------------------------------------------------------------------------
	# Create df_split, where divide the cysteines into invidiual residues
	#Use dictionary to make sure columns are the same
	#------------------------------------------------------------------------
	#FOR THE FIRST CYS
	
	
	
	
	df_  = pd.DataFrame(index = None)
	df_  =  pd.DataFrame({                                                
			"PDB"				:df['PDB'],                        
			"residue_number"	:df["residue_number"],                        
			"seq_length"		:df["seq_length"],                        
			"no_disulfides"		:df["no_disulfides"],                        
			"x1"				:df["X1"],                        
			"x1_array"  		:df["x1_array"],                            
			"x3"				:df["x3"],                                      
			"x3_array"			:df["x3_array"],                                      
			"Cb"				:df["CB"],                                 
			"Ca"				:df["CA"],                                 
			"Ha"				:df["HA"],                                 
			"Hn"				:df["HN"],                                 
			"N"					:df["N"],                                 
			"before Ca"			:df["before_CA"],                                 
			"after Ca"			:df["after_CA"],
			"before Ha"			:df["before_HA"],                                 
			"after Ha"			:df["after_HA"],
			"before Hn"			:df["before_HN"],                                 
			"after Hn"			:df["after_HN"],
			"ss_array"			:df["ss_array"]})		 		                                  
	
	
	
	df__  = pd.DataFrame(index = None)
	df__  =  pd.DataFrame({                                                
			"PDB"				:df['PDB'],                        
			"residue_number"	:df["residue_number"],
			"seq_length"		:df["seq_length"],                        
			"no_disulfides"		:df["no_disulfides"],                          
			"x1"				:df["X1"],                        
			"x1_array"  		:df["x1_array"],                            
			"x3"				:df["x3"],                                      
			"x3_array"			:df["x3_array"],                                        
			"Cb"				:df["CB"],                                 
			"Ca"				:df["CA"],                                 
			"Ha"				:df["HA"],                                 
			"Hn"				:df["HN"],                                 
			"N"					:df["N"],                                 
			"before Ca"			:df["before_CA"],                                 
			"after Ca"			:df["after_CA"],
			"before Ha"			:df["before_HA"],                                 
			"after Ha"			:df["after_HA"],
			"before Hn"			:df["before_HN"],                                 
			"after Hn"			:df["after_HN"],
			"ss_array"			:df["ss_array"]})		 		                                  
	
	
	df_split = df_.append(df__, ignore_index = True)
	
	
	final_database= pd.DataFrame([])#,columns = final_columns)
	
	cys_list=[]
	cys_list=(df_split['residue_number']).tolist()
	cys_list = list(set(cys_list))
	cys_list=sorted(cys_list, key=lambda x: float(x),reverse=True)
	possible_connections=[]
	possible_connections=list(itertools.combinations(cys_list, 2))
	print possible_connections
	
	
	for possible in possible_connections:
		possible    = list(possible)
		cys1,cys2   = possible[0], possible[1]
		
		cys1_shifts = df_split.loc[(df_split['residue_number'] == cys1)]
		cys2_shifts = df_split.loc[(df_split['residue_number'] == cys2)]
	
		cys1_shifts = cys1_shifts.reset_index(drop=True)
		cys2_shifts = cys2_shifts.reset_index(drop=True)
		
		joined		= pd.concat(dict(cys1_shifts = cys1_shifts, cys2_shifts = cys2_shifts),axis=1)
		#------------------------------------------------------------------------
		# IF PDB, CYS1 AND CYS2 ARE IN THE ORIGINAL DF, MEANS THEY ARE A TRUE CONNECTION
		#------------------------------------------------------------------------
	
		
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
	
	final_database = final_database[[
	"cys1_PDB",			
	"cys1_residue_number",
	"cys2_residue_number",
	"cys1_seq_length",
	"cys1_no_disulfides",
	"cys1_x1",			
	"cys1_x1_array",  	
	"cys1_x3",			
	"cys1_x3_array",			
	"cys1_Cb",			
	"cys1_Ca",			
	"cys1_Ha",			
	"cys1_Hn",			
	"cys1_N",				
	"cys1_before Ca",		
	"cys1_after Ca",		
	"cys1_before Ha",		
	"cys1_after Ha",		
	"cys1_before Hn",		
	"cys1_after Hn",		
	"cys1_ss_array",
	"cys2_x1",			
	"cys2_x1_array",  	
	"cys2_x3",			
	"cys2_x3_array",			
	"cys2_Cb",			
	"cys2_Ca",			
	"cys2_Ha",			
	"cys2_Hn",			
	"cys2_N",				
	"cys2_before Ca",		
	"cys2_after Ca",		
	"cys2_before Ha",		
	"cys2_after Ha",		
	"cys2_before Hn",		
	"cys2_after Hn",		
	"cys2_ss_array"]]
	
	final_database             = final_database.query ('cys1_x3 == cys2_x3')    
	final_database             = final_database.reset_index(drop=True)
	
	
	final_database.to_csv(peptide+'_connectivity.csv',index=False)	
	