import pandas as pd 
import numpy as np
import itertools
from pandas import Series
														
######################################
#------------------------------------
# To be used for creating hot arrays
#------------------------------------
######################################

def split_columns(dataframe,hot_array):
    length  = dataframe[hot_array][0]
    length  = len(length.split(','))
    s       = dataframe[hot_array].str.split('').apply(Series, 1).stack	()
    s.index = s.index.droplevel(-1)
    s.name  = hot_array
    del dataframe[hot_array]
    dataframe = dataframe.join(s.apply(lambda x: Series(x.split(','))))
    for i in range(length):
    	 x = str(hot_array)+str(i)
    	 dataframe = dataframe.rename(columns = {i:x})
    return dataframe

######################################
#------------------------------------
# Define "testing_input_list"
# This is ALL of  the required inputs AND THE CORRECT ORDER
# needed for the neural network. The generated dataframe (and csv file)
# will be generated in this order in the final command before return in this 
# script, whilst also defined in 'crosslink_neural_network' as another measure
# to ensure the order of inputs is the same as that of the saved model
#------------------------------------
######################################

testing_input_list = [
				   'PDB',
				   'cys1_residue',
				   'cys2_residue',				
                   'cys1_HA',
                   'cys2_HA',
                   'cys1_N',
                   'cys2_N',
                   'cys1_HN',
                   'cys2_HN',
                   'cys1_before_HN',
                   'cys2_before_HN',
                   'cys1_after_HN',
                   'cys2_after_HN',
                   'cys1_before_CA',
                   'cys2_before_CA',
                   'cys1_after_CA',
                   'cys2_after_CA',
                   'cys1_before_HA',
                   'cys2_before_HA',
                   'cys1_after_HA',
                   'cys2_after_HA',
                   'cys_diff',
                   'no_disulfides',
                   'cys1_ss_array0',
                   'cys1_ss_array1',
                   'cys1_ss_array2',
                   'cys2_ss_array0',
                   'cys2_ss_array1',
                   'cys2_ss_array2',
                   'cys1_x1_array0',
                   'cys1_x1_array1',
                   'cys1_x1_array2',
                   'cys2_x1_array0',
                   'cys2_x1_array1',
                   'cys2_x1_array2',
                   'cys1_x3_array0',
                   'cys1_x3_array1',
                   'cys2_x3_array0',
                   'cys2_x3_array1']


######################################
#------------------------------------
# The below function takes a list of individual cysteine residues
# Then generates a dataframe of ALL POSSIBLE isomers with associated inputs
# This inputs will be used in the neural network componenet of CrossLink
#------------------------------------
######################################

def generate_connectivity(df,peptide):
	
	######################################
	#------------------------------------
	# Initiate final_database that will store the isomers
	# Generate a list of all cysteine residues, then a 'possible_isomers' list 
	# that generates all possible pairings of cysteine residues
	#------------------------------------
	######################################		                                  
	
	final_database    = pd.DataFrame([])	
	cys_list          = []
	cys_list          = (df['residue']).tolist()
	cys_list          = list(set(cys_list))
	cys_list          = sorted(cys_list, key=lambda x: float(x),reverse=True)
	possible_isombers = []
	possible_isombers = list(itertools.combinations(cys_list, 2))
	
	print 'Possble isomers', possible_isombers
	
	######################################
	#------------------------------------
	# Iterate through each pairing and get the associated shifts
	# from each cysteine from the dataframe
	#------------------------------------
	######################################	
	
	for possible in possible_isombers:
		possible    = list(possible)
		cys1,cys2   = possible[0], possible[1]
		cys1_shifts = df.loc[(df['residue'] == cys1)]
		cys2_shifts = df.loc[(df['residue'] == cys2)]
		cys1_shifts = cys1_shifts.reset_index(drop=True)
		cys2_shifts = cys2_shifts.reset_index(drop=True)
		
		######################################
		#------------------------------------
		# Join the shifts into a single row
		# Append to the final dataframe
		#------------------------------------
		######################################	
		joined		   = pd.concat(dict(cys1_shifts = cys1_shifts, cys2_shifts = cys2_shifts),axis=1)
		final_database = final_database.append(pd.DataFrame(joined,index=[0]), ignore_index=True)
	
	######################################
	#------------------------------------
	# Rename the column names
	# Each column is an array with the name Eg.  ('cys1_shifts', 'HA'),  ('cys2_shifts', 'HA')
	# Rename the associated columns to be cys1_HA, cys2_HA
	#------------------------------------
	######################################	
	new_names    = []
	column_names = list(final_database.columns.values)
	print column_names
	for name in column_names:
		cys_number = name[0]
	
		if cys_number == 'cys1_shifts' or cys_number=='cys2_shifts':
			new_names.append(''.join([str(cys_number[0:5]), str(name[1])]))
		else:
			new_names.append(str(name[0]))
	
	
	######################################
	#------------------------------------
	# Define required structural inputs
	# Still need to inlcude number of disulfides, cys_diff etc
	#------------------------------------
	######################################

	final_database.columns = new_names
	final_database         = final_database[[
	"cys1_residue",
	"cys2_residue",
	"cys1_x1",			
	"cys1_x1_array",  	
	"cys1_x3",			
	"cys1_x3_array",			
	"cys1_HA",			
	"cys1_HN",			
	"cys1_N",				
	"cys1_before_CA",		
	"cys1_after_CA",		
	"cys1_before_HA",		
	"cys1_after_HA",		
	"cys1_before_HN",		
	"cys1_after_HN",		
	"cys1_ss_array",
	"cys2_x1",			
	"cys2_x1_array",  	
	"cys2_x3",			
	"cys2_x3_array",			
	"cys2_HA",			
	"cys2_HN",			
	"cys2_N",				
	"cys2_before_CA",		
	"cys2_after_CA",		
	"cys2_before_HA",		
	"cys2_after_HA",		
	"cys2_before_HN",		
	"cys2_after_HN",		
	"cys2_ss_array"]]
	
	######################################
	#------------------------------------
	# Search database for cysteine pairings where the x3 angles are the same
	# Cysteines that are bonded share the same x3 angle, therefore will have the same value
	#------------------------------------
	######################################	
	final_database = final_database.query ('cys1_x3 == cys2_x3')    
	final_database = final_database.reset_index(drop=True)
	
	######################################
	#------------------------------------
	# Add in the PDB, no disulfides and cys_diff
	#------------------------------------
	######################################
	final_database['PDB'          ] = peptide
	final_database['no_disulfides'] = df['no_disulfides']
	final_database['cys_diff'     ] = final_database['cys1_residue'] - final_database['cys2_residue']
	

	######################################
	#------------------------------------
	# Split the hot_arrays
	#------------------------------------
	######################################
	final_database = split_columns(final_database,'cys1_ss_array')
	final_database = split_columns(final_database,'cys2_ss_array')
	final_database = split_columns(final_database,'cys1_x1_array')
	final_database = split_columns(final_database,'cys2_x1_array')
	final_database = split_columns(final_database,'cys1_x3_array')
	final_database = split_columns(final_database,'cys2_x3_array')

	######################################
	#------------------------------------
	# Ensure database order is the same as that used to train the network (testing_input_list)
	# Save and return dataframe
	#------------------------------------
	######################################
	final_database = final_database[testing_input_list]   
	final_database.to_csv(peptide+'_connectivity_inputs.csv',index = 'False')
	return(final_database)
	