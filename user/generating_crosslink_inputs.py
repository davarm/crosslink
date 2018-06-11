import os
from glob import glob
import itertools
import pandas as pd 
import numpy as np
from collections import OrderedDict
import os,sys

# Script to generate the required inputs for the CrossLink program
# Chemical shifts , sequence and secondary are extracted from TALOS-N files (README)
# The program does not predict termini cysteines and therefore they are excluded
# If chemical shifts are unassigned they are designated as '0'
#
#				: ./generating_DISH_inputs.py 2n8e
#
#

peptide = sys.argv[1]
path = "./peptides/"+peptide


#----------------------------------
#----------------------------------
# Read the sequence and identify all oxidised Cys residues
# NOTE: OXIDISED CYS RESIDUES MUST BE LOWERCASE 'c' IN THE TALOS .SEQ FILE
#----------------------------------
#----------------------------------

with open(path+'/'+peptide+'.seq') as f:
	sequence = f.readline()

cys_residues = [i for i, x in enumerate(sequence) if x == "c"]
cys_residues = [i+1 for i in cys_residues]


sequence_dict = {}
for k,residue in enumerate(sequence):
		sequence_dict[str(k+1)            ] = residue
		sequence_dict['0'                 ] = 'TERMINAL'
		sequence_dict[str(len(sequence)+1)] = 'TERMINAL'

#------------------------------------------------------------------------
# Create ordered dictionaries to store all relevant information
#------------------------------------------------------------------------
cysteine_dict            =  OrderedDict()
nuclei_list = ['HA','CB','CA','HN','N']
before_nuclei_list = ['HA','CA','HN']

cysteine_dict['PDB'    ] = []
cysteine_dict['residue'] = []
cysteine_dict['ss_array' ] = []
	
for nuclei in nuclei_list:
	cysteine_dict[nuclei] = [] 

for nuclei in before_nuclei_list:	
	cysteine_dict['before_' + nuclei] = []
	cysteine_dict['after_' + nuclei] = []
	

# Initiate dataframe to store information
df = pd.DataFrame()



#--------------------------------------------------------------------------------
# Storing chemical shifts by residue number and nuclei in experimenatl_shift dict
# Chemical shifts stored in the TALOS-N format '.tab' file
# Store in the dicitonary as a combination of residue number and nuclei type
#---------------------------------------------------------------------------------

adjusted_chemical_shift_dict = {}
get                          = open(path+'/predAdjCS.tab','r') 
for line in get:
		line  = " ".join(line.split())
		if 'REMARK' in line:
			continue 
		if 'VARS' in line:
			continue
		if 'VARS' in line:
			continue
		if 'FORMAT' in line:
			continue 
		if len(line) == 0:
			continue
		if 'DATA SEQUENCE' in line:
			continue

		# Take chemical shifts and strore into dictionary 		
		lines = line.split(' ')
		try:
			adjusted_chemical_shift_dict[lines[0]+','+lines[2]]=lines[3]
		except IndexError:
			print 'error', line
	


#----------------------------------
# Add In secondary structure
#----------------------------------
ss_dict = {}
secondary_structure_list = [
	'H',
	'E',
	'L'] 
#print ss

get                          = open(path+'/predSS.tab','r') 
for line in get:
		line  = " ".join(line.split())
		if 'REMARK' in line:
			continue 
		if 'VARS' in line:
			continue
		if 'DATA' in line:
			continue
		if 'FORMAT' in line:
			continue 
		if len(line) == 0:
			continue
		line = line.split(' ')
		ss_hot_array 									 = [0. for _ in range(3)]  
		ss_hot_array[secondary_structure_list.index(line[8])] = 1
		ss_hot_array                                     = [str(x) for x in ss_hot_array]
		ss_hot_array                                     = ",".join(ss_hot_array)
		ss_dict[line[0]] = ss_hot_array
#---------------------------------------------------------------------------------
# All of the shift and secondary structure information has been stored in dictionaries
# Can now go through each connectivity in the previously stored 'Connectivity list'
# Start to assemble information
#---------------------------------------------------------------------------------

for cysteine in cys_residues:
			if cysteine == 1 or cysteine == len(sequence):
				continue
			cys_before  = str(cysteine-1)
			cys_after  = str(cysteine+1)
			cysteine    = str(cysteine)

			#------------------------------------------------------------------------
			# CHEMICAL SHIFTS FOR CYSTEINES 
			#------------------------------------------------------------------------


			#------------------------------------------------------------------------
			# IF CHEMICAL SHIFT ISN"T ASSIGNED RETURN AS ZERO
			#------------------------------------------------------------------------

			def chemical_shift(residue,nuclei):
				try:
					value = adjusted_chemical_shift_dict[residue+','+nuclei]
				except KeyError:
					value =  0
				return (value)

			#------------------------------------------------------------------------
			# STORE IN DICT
			#------------------------------------------------------------------------

			for nuclei in nuclei_list:
				cysteine_dict[nuclei].append(chemical_shift(cysteine,nuclei       ))
			
			for nuclei in before_nuclei_list:
				cysteine_dict['before_'+nuclei].append(chemical_shift(cys_before,nuclei))
				cysteine_dict['after_'+nuclei].append(chemical_shift(cys_before,nuclei))
			#------------------------------------------------------------------------
			# Adding in residues before
			#------------------------------------------------------------------------
			
			cysteine_dict['ss_array'].append(ss_dict[cysteine])
			#------------------------------------------------------------------------
			# Add in PDB and residues
			#------------------------------------------------------------------------
			cysteine_dict['PDB'].append(peptide)
			cysteine_dict['residue'].append(cysteine)

#------------------------------------------------------------------------
# ADD all stored in the cysteine dict to DF
#------------------------------------------------------------------------

for key in cysteine_dict:
	xx = np.array(cysteine_dict[key])
	df[key]=	(xx)
df['no_disulfides'] = len(cys_residues) / 2
df['x1'] = -60
df.to_csv(peptide+'_crosslink.csv',index=False)
print df
print 'MUST DO: PLEASE MANUALLY ENTER X1 ANGLES: EITHER +60, -60, 180'

