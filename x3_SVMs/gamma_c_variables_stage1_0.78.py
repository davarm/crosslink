import sys

#import shuffle
from sklearn.metrics import *
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from random import shuffle
import numpy as np
import pandas as pd
from pandas import Series

df          = pd.read_csv('Cys_database.csv',sep = ',', skipinitialspace = False)

df = df.loc[df['x1'] == (-60)]
df = df.reset_index(drop=True)
# stage1_df = df.loc[df['x1'] == (-60)]


inputs = [			'PDB',
					'x3',
					# 'cys1_psi',
					'cys1_ss_array0',
					'cys1_ss_array1',
					'cys1_ss_array2',
					'cys1_Ha',
					'cys1_Ca',
					'cys1_Cb',
					#'cys1_N',
					# 'cys1_Hn',
					'cys1_before Ha',
					# 'cys1_after Ha',
					'cys1_before Ca']
					# 'cys1_after Ca']
#------------------------------------------------------------------------
##HAVE TO CONVERT THE SS ARRAY FROM A STRING TO SEPERATE COLUMNS IN THE DATABSE
##------------------------------------------------------------------------
def stage_1_split(x3):
	x3   = str(x3)
	if x3 == '-90.0':
		x3_index = 0
	if x3 == '90.0':
		x3_index = 1
	return x3_index

df['x3'] =  df['x3'].apply(stage_1_split)

def split_columns(dataframe,hot_array):
	 # print hot_array
	 #print dataframe[hot_array][0]
	 length=dataframe[hot_array][0]
	 length=len(length.split(','))
	 s                     = dataframe[hot_array].str.split('').apply(Series, 1).stack	()
	 s.index               = s.index.droplevel(-1)
	 s.name                = hot_array
	 del dataframe[hot_array]
	 dataframe =dataframe.join(s.apply(lambda x: Series(x.split(','))))
	 for i in range(length):
		 x=str(hot_array)+str(i)
		 dataframe=dataframe.rename(columns = {i:x})
	 return dataframe
df = split_columns(df,'cys1_ss_array')
df = split_columns(df,'x1_array')
df = split_columns(df,'cys1_b_residue_array')
df =df[inputs]
# print df.dtypes.tolist()
#------------------------------------------------------------------------
# Create seperate PDB lists to cycle through for testing
#------------------------------------------------------------------------
pdb_list   = df['PDB'].tolist()
pdb_list   = list(set(pdb_list))



##------------------------------------------------------------------------
## START OFF TESTING -60 FIRST (SVM-1)
##------------------------------------------------------------------------
y_actual = []
y_pred = []
truex=[]
max_list=[]
#y_actual_2 = []
#y_pred_2 = []
#truex_2=[]
#
##Generate Training inputs and target
a=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
b=[9000,8000,7000,6000,5000,4000,3000,1000,500,400,300,200,100,90,80,70,60,50,40,30,20,10,9,8,7,6,5,4,3,2,1]

import itertools
possible=[]
for r in itertools.product(a, b): 
  possible.append(r)

for combo in possible:
	y_actual = []
	y_pred = []
	
	for pdb in pdb_list:

		training_df=[]
		testing_df=[]
		training_target=[]
		testing_target=[]
	
		training_df     = df.loc[df['PDB'] != (pdb)]
		training_df     = training_df.reset_index(drop=True)
		training_target = training_df['x3']
		training_df     = training_df.drop('PDB',1)
		training_df     = training_df.drop('x3',1)
		#training_df = split_columns(training_df,'cys1_ss_array')
		# training_df = split_columns(training_df,'cys1_b_residue_array')
		# training_df = split_columns(training_df,'cys1_a_residue_array')
		training_df     = training_df.astype(float)

		#print training_df.dtypes
		
		#Generate testing inputs and target
		testing_df      = df.loc[df['PDB'] == (pdb)]
		testing_df      = testing_df.reset_index(drop=True)
		testing_target  = testing_df['x3']
		testing_df      = testing_df.drop('PDB',1)
		testing_df      = testing_df.drop('x3',1)
		#testing_df = split_columns(testing_df,'cys1_ss_array')
		# testing_df = split_columns(testing_df,'cys2_ss_array')
		# testing_df = split_columns(testing_df,'cys1_b_residue_array')
		# testing_df = split_columns(testing_df,'cys1_a_residue_array')
		# testing_target = testing_target.reset_index(drop=True)
		# scaler = preprocessing.RobustScaler().fit(training_df)
		scaler = preprocessing.MinMaxScaler().fit(training_df)
		# scaler = preprocessing.StandardScaler().fit(training_df)

		training_scaled =scaler.transform(training_df)
		testing_scaled = scaler.transform(testing_df)
	
		clf=svm.SVC(kernel='rbf',gamma=float(combo[0]),C=float(combo[1]))		
		
		#Non Scaled
		# clf.fit(training_df,training_target)
		# prediction     = clf.predict(testing_df)
		
		#SCALED
		clf.fit(training_scaled,training_target)
		prediction     = clf.predict(testing_scaled)
	
	
		for index,_ in enumerate(prediction):
			y_actual.append(testing_target[index])
			y_pred.append(_)

	print len(y_pred)
	print combo,matthews_corrcoef(y_actual, y_pred) 
	max_list.append(matthews_corrcoef(y_actual, y_pred))

print 'MAX SCORE',max(max_list)
print 'Min SCORE',min(max_list)
