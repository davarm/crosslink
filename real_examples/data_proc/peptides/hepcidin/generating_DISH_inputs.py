import os
from glob import glob
import itertools
import pandas as pd 
import numpy as np
from collections import OrderedDict
import os,sys

# Script to generate the required inputs for the DISH program
# Chemical shift, sequence and backbone angles are extracted from TALOS-N files (README)
# Must have at minimum the 'pred.tab' and 'predAdjCS.tab' files
# These should NOT be changed. The sequence is extracted from the pred.tab file
# Must also have an additional 'connectivity.txt' file that defines correct cysteine pairings 
# The program does not predict termini cysteines and therefore they are excluded
# If chemical shifts are unassigned they are designated as '0'
# 'hemi' means an input from the corresponding hemi-cysteine of a disulfide bond (as derived from connectivity.txt)
# Backbone angles are only taken from TALOS-N if they are 'STRONG or GENEROUS' predictions
# If they are 'None' or 'Warn', inputs are substituted with the average phi and psi angles derived from the training database.
# To run specifiy folder name as first argv 
#
#				: ./generating_DISH_inputs.py 2n8e
#
#
path = "."
peptide = "hepecidin"
#
#------------------------------------------------------------------------
# Create ordered dictionaries to store all relevant information
#------------------------------------------------------------------------
nuclei_list   = ['HA','CB','CA','N','HN']
cysteine_dict =  OrderedDict()

for nuclei in nuclei_list:
	cysteine_dict['Cys1_'        + nuclei] = [] 
	cysteine_dict['Cys2_'        + nuclei] = []
	cysteine_dict['Cys1_after_'  + nuclei] = []
	cysteine_dict['Cys2_after_'  + nuclei] = []
	cysteine_dict['Cys1_before_' + nuclei] = []
	cysteine_dict['Cys2_before_' + nuclei] = []
	
	
cysteine_dict['PDB'                ] = []
cysteine_dict['Cys1_phi'           ] = []
cysteine_dict['Cys1_psi'           ] = []
cysteine_dict['Cys1_SS'           ] = []
cysteine_dict['Cys2_SS'           ] = []

cysteine_dict['Cys2_phi'           ] = []
cysteine_dict['Cys2_psi'           ] = []
cysteine_dict['residue_1'          ] = []
cysteine_dict['residue_2'          ] = []
cysteine_dict['Cys1_after_psi'     ] = []
cysteine_dict['Cys1_after_phi'     ] = []
cysteine_dict['Cys2_after_phi'     ] = []
cysteine_dict['Cys2_after_psi'     ] = []
cysteine_dict['Cys1_before_phi'    ] = []
cysteine_dict['Cys1_before_psi'    ] = []
cysteine_dict['Cys2_before_phi'    ] = []
cysteine_dict['Cys2_before_psi'    ] = []
cysteine_dict['Cys1_after_residue' ] = []
cysteine_dict['Cys2_after_residue' ] = []
cysteine_dict['Cys1_before_residue'] = []
cysteine_dict['Cys2_before_residue'] = []
	


# Initiate dataframe to store information
df = pd.DataFrame()


#---------------------------------------------------
# Defining connetivity in a 'connectivity.txt' file
# Store in connectivity_list
#---------------------------------------------------
get               = open(path+'/connectivity.txt','r')
chi1_dict         = {}
cys_list          = []
connectivity_list = []
for line in get:
	line = line.strip('\n')
	if ':' in line:
			line = line.split(':')
			cys_list.append(line[0])
			cys_list.append(line[1])
			connectivity_list.append(line[0:2])
get.close()        


#--------------------------------------------------------------------------------
# Storing chemical shifts by residue number and nuclei in experimenatl_shift dict
# Chemical shifts stored in the TALOS-N format '.tab' file
# Store in the dicitonary as a combination of residue number and nuclei type
#Calculating secondary shift values is done later on to simplify process as must 
#accommodate for Proline residues changing RC values
#---------------------------------------------------------------------------------

adjusted_chemical_shift_dict = {}
sequence                     = []
sequence_dict                = {}
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

		# In TALOS-N sequence can be spread over two lines
		# Will convert and append into a singular line
		if 'DATA SEQUENCE' in line and len(sequence) != 0:
				sequence2 = line.split(' ')
				sequence2 = sequence2[2:]
				sequence2 = "".join(sequence2)
				sequence  = sequence + sequence2

		
		
		# EXTRACTING THE SEQUENCE FROM THE 'DATA SEQUENCE LINE'
		if 'DATA SEQUENCE' in line and len(sequence) == 0:
				sequence = line.split(' ')
				sequence = sequence[2:]
				sequence = "".join(sequence)
		# TAke chemical shifts and strore into dictionary 
		
		lines = line.split(' ')
		try:
			adjusted_chemical_shift_dict[lines[0]+','+lines[2]]=lines[3]
		except IndexError:
			print 'error', line
	

#----------------------------
# Storing sequence information into a dictionrary
# Residue number is the key, residue the value
# Add terminal to each
#----------------------------

for k,residue in enumerate(sequence):
		sequence_dict[str(k+1)            ] = residue
		sequence_dict['0'                 ] = 'TERMINAL'
		sequence_dict[str(len(sequence)+1)] = 'TERMINAL'



#---------------------------------------------------------------------	
# Gaining the phi and psi angles from TALOS-N results 'pred.tab'
# Store in a dictionary based on residue number (storing all residues)
# If TALOS-N does not make a prediction store as 'NaN'
#---------------------------------------------------------------------	

phi_dict      = {}
psi_dict      = {}
phi_dict['0'] = 'NaN'
psi_dict['0'] = 'NaN'

phi_dict[str(len(sequence)+1)] = 'N/A'
psi_dict[str(len(sequence)+1)] = 'N/A'
get           = open(path+'/pred.tab','r')

for line in get:
		line  = " ".join(line.split())
		lines = line.split(' ')
		if 'REMARK' in line:
			continue 
		if 'VARS' in line:
			continue
		if 'FORMAT' in line:
			continue
		if 'DATA' in line:
			continue  
		if len(line) == 0:
			continue


		### YOU CAN  CHOOSE TO COMMENT OUT HERE WHAT TALOS_N BACKBONE PREDICTIONS THAT YOU IGNORE
		if lines[10] == 'None'  or lines[10] == 'Dyn':#or lines[10] == 'Warn'
 			phi_dict[lines[0]] = 'N/A'
			psi_dict[lines[0]] = 'N/A'
		else:  
			phi_dict[lines[0]] = lines[2]
			psi_dict[lines[0]] = lines[3]
get.close()   


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
#All of the shift and angle information has been stored in dictionaries
#Can now go through each connectivity in the previously stored 'Connectivity list'
#Start to assemble information
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------

#								REQUIRED FORMAT FOR DATABASE

# Nuclei order = [N,HA,C,CA,CB,HN]

# Peptide name: Chain: Cys1: Chain: 
# Cys2: Cys1 Nuclei: Cys2 Nuclei:
# X1,X2,X3,X2',X1'
# Cys1 Phi,Psi,X1  
# Cys2 Phi,Psi,X1

#  NEIGHBOURING INFORMATION For Cys1
# Cys1_before Nuclei
# Cys1_before Phi,Psi,X1
# Cys1_before residue type

# Cys1_after Nuclei
# Cys1_after Phi,Psi,X1
# Cys1_after residue type--

# NEI2HBOURING INFORMATION For Cys1
# Cys2_before Nuclei
# Cys2_before Phi,Psi,X1
# Cys2_before residue type

# Cys2_after Nuclei
# Cys2_after Phi,Psi,X1
# Cys2_after residue type
#---------------------------------------------------------------------------------


#For each connectivity
for connectivity in connectivity_list:
			connectivity = [int(x) for x in connectivity]
			#---------------------------------------------
			#By default, the larger residue number is Cys1
			#---------------------------------------------
			connectivity = sorted(connectivity, key=int, reverse=True) 
			cys1         = str(connectivity[0])
			cys2         = str(connectivity[1])
			cys1_before  = str(int(cys1)-1    )
			cys1_after   = str(int(cys1)+1    )
			cys2_before  = str(int(cys2)-1    )
			cys2_after   = str(int(cys2)+1    )

			#------------------------------------------------------------------------
			# CHEMICAL SHIFTS FOR CYSTEINES 
			#------------------------------------------------------------------------
			nuclei_list = ['HA','CB','CA','N','HN']


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

				cysteine_dict['Cys1_'       +nuclei].append(chemical_shift(cys1,nuclei       ))
				cysteine_dict['Cys2_'		+nuclei].append(chemical_shift(cys2,nuclei       ))
				cysteine_dict['Cys1_before_'+nuclei].append(chemical_shift(cys1_before,nuclei))
				cysteine_dict['Cys2_before_'+nuclei].append(chemical_shift(cys2_before,nuclei))
				cysteine_dict['Cys1_after_' +nuclei].append(chemical_shift(cys1_after,nuclei ))
				cysteine_dict['Cys2_after_' +nuclei].append(chemical_shift(cys2_after,nuclei ))

			#------------------------------------------------------------------------
			# Adding in the Backbones
			#------------------------------------------------------------------------
		
			cysteine_dict['Cys1_phi'       ].append(phi_dict[cys1		])
			cysteine_dict['Cys1_psi'       ].append(psi_dict[cys1		])
			cysteine_dict['Cys2_phi'       ].append(phi_dict[cys2		])
			cysteine_dict['Cys2_psi'       ].append(psi_dict[cys2		])
			cysteine_dict['Cys1_before_phi'].append(phi_dict[cys1_before])
			cysteine_dict['Cys1_before_psi'].append(psi_dict[cys1_before])
			cysteine_dict['Cys2_before_phi'].append(phi_dict[cys2_before]) 
			cysteine_dict['Cys2_before_psi'].append(psi_dict[cys2_before])
			cysteine_dict['Cys1_after_phi' ].append(phi_dict[cys1_after	])
			cysteine_dict['Cys1_after_psi' ].append(psi_dict[cys1_after	])
			cysteine_dict['Cys2_after_phi' ].append(phi_dict[cys2_after	])
			cysteine_dict['Cys2_after_psi' ].append(psi_dict[cys2_after	])

			#------------------------------------------------------------------------
			# Adding in residues before
			#------------------------------------------------------------------------
			
			cysteine_dict['Cys1_SS'].append(ss_dict[cys1])
			cysteine_dict['Cys2_SS'].append(ss_dict[cys2])
						
			#------------------------------------------------------------------------
			# Adding in residues before
			#------------------------------------------------------------------------
			
			cysteine_dict['Cys1_before_residue'].append(sequence_dict[cys1_before])
			cysteine_dict['Cys1_after_residue' ].append(sequence_dict[cys1_after ])
			cysteine_dict['Cys2_before_residue'].append(sequence_dict[cys2_before])
			cysteine_dict['Cys2_after_residue' ].append(sequence_dict[cys2_after ])
			
			#------------------------------------------------------------------------
			# Add in PDB and residues
			#------------------------------------------------------------------------
			cysteine_dict['PDB'].append(peptide)
			cysteine_dict['residue_1'].append(cys1)
			cysteine_dict['residue_2'].append(cys2)

#------------------------------------------------------------------------
# ADD all stored in the cysteine dict to DF
#------------------------------------------------------------------------

for key in cysteine_dict:
	xx = np.array(cysteine_dict[key])
	df[key]=	(xx)





#----------------------------------
# Take into account
#----------------------------------

#----------------------------------
# SPLIT INTO INDIVIDUAL HEMI-CYSTEINES, WITH INFORMATION OF CORRESPONDING HEMI-CYSTEINE REQUIRED FOR DISH INPUTS
#----------------------------------

new_columns = ["PDB",				
"residue_number",	
"N",					
"HN",				
"HA",				
"CA",				
"CB",				
"before_CA",
"after_CA",
"before_HA",
"after_HA",
"before_HN",
"after_HN",
"ss_array"]


df_  = pd.DataFrame(index = None,columns = new_columns)
df_  =  pd.DataFrame({                                                
			"PDB"				:df['PDB'                    ],
			"residue_number"	:df["residue_1"              ],
			"N"					:df["Cys1_N"                 ],
			"HN"				:df["Cys1_HN"                ],
			"HA"				:df["Cys1_HA"                ],
			"CA"				:df["Cys1_CA"                ],
			"CB"				:df["Cys1_CB"                ],
			"before_CA"			:df["Cys1_before_CA"         ],
			"after_CA"			:df["Cys1_after_CA"          ],
			"before_HN"			:df["Cys1_before_HN"         ],
			"after_HN"			:df["Cys1_after_HN"          ],
			"before_HA"			:df["Cys1_before_HA"         ],
			"after_HA"			:df["Cys1_after_HA"          ],
			"ss_array"			:df["Cys1_SS"          ],
			})
	
	
	
df__ = pd.DataFrame(index = None,columns = new_columns)
df__  =  pd.DataFrame({                                                
			"PDB"				:df['PDB'                    ],
			"residue_number"	:df["residue_2"              ],
			"N"					:df["Cys2_N"                 ],
			"HN"				:df["Cys2_HN"                ],
			"HA"				:df["Cys2_HA"                ],
			"CA"				:df["Cys2_CA"                ],
			"CB"				:df["Cys2_CB"                ],
			"before_CA"			:df["Cys2_before_CA"         ],
			"after_CA"			:df["Cys2_after_CA"          ],
			"before_HN"			:df["Cys2_before_HN"         ],
			"after_HN"			:df["Cys2_after_HN"          ],
			"before_HA"			:df["Cys2_before_HA"         ],
			"after_HA"			:df["Cys2_after_HA"          ],
			"ss_array"			:df["Cys2_SS"          ],
			})
	
	

df_split = df_.append(df__, ignore_index = True)


df_split = df_split
[["PDB"				
"residue_number",	
"N",					
"HN",				
"HA",				
"CA",				
"CB",				
"before_CA",
"after_CA",
"before_HA",
"after_HA",
"before_HN",
"after_HN",
"ss_array"]]



#----------------------------------
#
# IF TALOS-N DOES NOT MAKE A PREDICTION FOR BACKBONE ANGLES, SUBSTITUTE WITH 0s.
# Psi average = +75
# Phi Average = -92
# Before_psi_average = + 63
#----------------------------------



print df_split
df_split                        = df_split.replace(r'\s+', np.nan, regex=True)
df_split = df_split[new_columns]
df_split.to_csv(path+'/DISH_inputs.csv',index = False)
			# print df 
