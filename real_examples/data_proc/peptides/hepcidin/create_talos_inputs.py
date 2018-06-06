#----------------------------------
# Have to convert the _cs.str chemical shift files to talos.tab format
#----------------------------------
import os
from glob import glob


#xx=glob("./new_peptides/*/")
#peptide_list=glob("./new_peptides/*/")
#for peptide in peptide_list:
#	print peptide
#	path         = peptide
#	peptide_name = peptide[-5:-1]
#	print 		  peptide_name

peptide_name = 'hepcidin'
#----------------------------------
# Read and store the sequence 
# Store as peptide.seq
#----------------------------------

with open(peptide_name+'.fasta.txt') as f:
	content = f.readlines()
sequence = content[1]
sequence = sequence.replace('C','c')
get = open(peptide_name+'.seq','w')

get.write(str(sequence))
get.close()

#----------------------------------
# Read the _cs.str file. Store as .tab
#----------------------------------
chemical_shifts = []
with open(peptide_name+'_cs.str', 'r') as f:
	for line in f:
		line = " ".join(line.split())
		if '_Atom_chem_shift.Assigned_chem_shift_list_ID' in line:              
			print line 
			for k,line in enumerate(f):
				if k ==0:
					continue
				# print line
				if 'stop' in line:
					break
				if k != 0:
					line = line.strip('\n')
					line = line.replace('CYS','cys')
					# line = line.replace('H','HN')
					line = ' '.join(line.split())
					line = line.split(' ')
					chemical_shifts.append([line[5]]+[line[6]]+[line[7]]+[line[10]])

#----------------------------------
# Write out the .tab file
#----------------------------------
get = open(peptide_name+'.tab','w')
get.write('DATA FIRST_RESID 1'+'\n')
get.write('DATA SEQUENCE '+str(sequence))
get.write('\n')
get.write('\n')
get.write('VARS RESID RESNAME ATOMNAME SHIFT'+'\n')
get.write('FORMAT %4d   %1s     %4s      %8.3f'+'\n')
get.write('\n')

for string in chemical_shifts:
	# string = string.replace('CYS','cys')
	if string[2] == 'H':
		string[2]=  'HN'
	get.write('{:4d} {:1s} {:4s} {:8.3f}'.format(int(string[0]),string[1],string[2],float(string[3])))
	get.write('\n')
