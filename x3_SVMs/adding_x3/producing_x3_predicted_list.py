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
# from keras.models import Sequential
# from keras.layers import Dense

df          = pd.read_csv('./../Cys_database.csv',sep = ',', skipinitialspace = False)


def split_columns(dataframe,hot_array):
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

df=split_columns(df,'cys1_ss_array')

df=split_columns(df,'x1_array')

minus60_df = df.loc[df['x1'] == (-60)]
other_df   = df.loc[df['x1'] != (-60)]
minus60_df = minus60_df.reset_index(drop=True)
other_df   = other_df.reset_index(drop=True)

#For 'other' testing the inputs required are 'Ca, Cb, x1_hot_array, ss_array'
# SVM_1_df   = minus60_df[['PDB','Ha','Ca','Cb','before Ca','psi','phi','ss_array','x3']]
SVM_1_df   = minus60_df[['PDB',
					'cys1_residue',
					'x3',
					'cys1_ss_array0',
					'cys1_ss_array1',
					'cys1_ss_array2',
					'cys1_Ha',
					'cys1_Ca',
					'cys1_Cb',
					'cys1_before Ha',
					'cys1_before Ca']]
	
SVM_2_df   =  other_df[['PDB',
					'cys1_residue',
					'x3',
					'x1_array0',
					'x1_array1',
					'x1_array2',
					'cys1_Ha',
					'cys1_Ca',
					'cys1_Cb']]
#------------------------------------------------------------------------
#HAVE TO CONVERT THE SS ARRAY FROM A STRING TO SEPERATE COLUMNS IN THE DATABSE
#------------------------------------------------------------------------


# SVM_2_df=spblit_columns(SVM_2_df,'ss_array')
#------------------------------------------------------------------------
# Create seperate PDB lists to cycle through for testing
#------------------------------------------------------------------------
SVM_1_pdb_list   = SVM_1_df['PDB'].tolist()
SVM_1_pdb_list   = list(set(SVM_1_pdb_list))

SVM_2_pdb_list   = SVM_2_df['PDB'].tolist()
SVM_2_pdb_list   = list(set(SVM_2_pdb_list))

# print SVM_1_df
#------------------------------------------------------------------------
# START OFF TESTING -60 FIRST (SVM-1)
#------------------------------------------------------------------------
y_actual = []
y_pred = []
truex=[]
y_actual_2 = []
y_pred_2 = []
truex_2=[]
config =[-90,90]
#Generate Training inputs and target
def SVM_1_testing(pdb):
	training_df=[]
	testing_df=[]
	training_target=[]
	testing_target=[]

	training_df     = SVM_1_df.loc[SVM_1_df['PDB'] != (pdb)]
	training_df     = training_df.reset_index(drop=True)
	training_target = training_df['x3']
	training_df     = training_df.drop('PDB',1)
	training_df     = training_df.drop('cys1_residue',1)
	training_df     = training_df.astype(float)
	training_df     = training_df.drop('x3',1)
	#print training_df.dtypes
	
	#Generate testing inputs and target
	testing_df      = SVM_1_df.loc[SVM_1_df['PDB'] == (pdb)]
	testing_df      = testing_df.reset_index(drop=True)
	residue_number  = testing_df['cys1_residue'].tolist()
	testing_target  = testing_df['x3']


	testing_df      = testing_df.drop('PDB',1)
	testing_df      = testing_df.drop('cys1_residue',1)
	testing_df      = testing_df.drop('x3',1)
	scaler = preprocessing.MinMaxScaler().fit(training_df)
	# scaler = preprocessing.RobustScaler().fit(training_df)
	training_scaled =scaler.transform(training_df)
	# clf=svm.SVC(kernel='rbf',gamma=0.5,C=20)
	clf=svm.SVC(kernel='rbf',gamma=0.4,C=20)
	clf.fit(training_scaled,training_target)

	testing_inputs = scaler.transform(testing_df)
	prediction     = clf.predict(testing_inputs)
	# print prediction
	for index,_ in enumerate(prediction):
		# print _
		print pdb,',',residue_number[index],',',_
		y_actual.append(testing_target[index])
		y_pred.append(_)

		if _ == testing_target[index]:
			truex.append('x')

	return(y_actual,y_pred)

for _ in SVM_1_pdb_list:
	SVM_1_testing(_)

print 'SVM1 accuracy'
print len(y_actual)
print matthews_corrcoef(y_actual, y_pred) 
print accuracy_score(y_actual, y_pred) 
print len(truex)

def SVM_2_testing(pdb):
	training_df=[]
	testing_df=[]
	training_target=[]
	testing_target=[]

	training_df     = SVM_2_df.loc[SVM_2_df['PDB'] != (pdb)]
	training_df     = training_df.reset_index(drop=True)
	training_target = training_df['x3']
	training_df     = training_df.drop('PDB',1)
	training_df     = training_df.drop('cys1_residue',1)
	training_df     = training_df.astype(float)
	training_df     = training_df.drop('x3',1)
	#print training_df.dtypes
	
	#Generate testing inputs and target
	testing_df      = SVM_2_df.loc[SVM_2_df['PDB'] == (pdb)]
	testing_df      = testing_df.reset_index(drop=True)
	testing_target  = testing_df['x3']


	residue_number  = testing_df['cys1_residue'].tolist()
	testing_df      = testing_df.drop('PDB',1)
	testing_df     = testing_df.drop('cys1_residue',1)
	testing_df      = testing_df.drop('x3',1)

	scaler = preprocessing.StandardScaler().fit(training_df)
	training_scaled =scaler.transform(training_df)


	clf=svm.SVC(kernel='rbf',gamma=0.5,C=1)
	clf.fit(training_scaled,training_target)

	testing_inputs = scaler.transform(testing_df)
	prediction     = clf.predict(testing_inputs)
	for index,_ in enumerate(prediction):

		print pdb,',',residue_number[index],',',_
		y_actual_2.append(testing_target[index])
		y_pred_2.append(_)

		if _ == testing_target[index]:
			truex_2.append('x')

	return(y_actual_2,y_pred_2)

for _ in SVM_1_pdb_list:
	SVM_1_testing(_)

for _ in SVM_2_pdb_list:
	SVM_2_testing(_)


print 'SVM2 Accuracy'
print matthews_corrcoef(y_actual_2, y_pred_2) 
print accuracy_score(y_actual_2, y_pred_2) 
print len(truex_2)