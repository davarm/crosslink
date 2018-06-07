import sys
from sklearn.metrics import *
from sklearn import svm
from sklearn import preprocessing
from random import shuffle
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.externals import joblib
import pickle

#----------------------------------
#----------------------------------
# Read Cys database
#----------------------------------
#----------------------------------

df = pd.read_csv('Cys_database.csv',sep = ',', skipinitialspace = False)

#----------------------------------
#----------------------------------
#  Define function that will split arrays into individual columns
#----------------------------------
#----------------------------------

def split_columns(dataframe,hot_array):
			length                 = dataframe[hot_array][0]
			length                 = len(length.split(','))
			s                      = dataframe[hot_array].str.split('').apply(Series, 1).stack	()
			s.index                = s.index.droplevel(-1)
			s.name                 = hot_array
			del dataframe[hot_array]
			dataframe              = dataframe.join(s.apply(lambda x: Series(x.split(','))))
			for i in range(length):
			x                      = str(hot_array)+str(i)
			dataframe              = dataframe.rename(columns = {i:x})
	 		return dataframe

#----------------------------------
#----------------------------------
# Split the secondary structure and x1 arrays
#----------------------------------
#----------------------------------

df = split_columns(df,'cys1_ss_array')
df = split_columns(df,'x1_array')

#----------------------------------
#----------------------------------
# The SVM-X3 splits Cys residues into two groups:
#	X1 == -60         (SVM_1_df)
#   X1 == +60 or 180  (SVM_2_df)
#----------------------------------
#----------------------------------

SVM_1_df = df.loc[df['x1'] == (-60)]
SVM_2_df = df.loc[df['x1'] != (-60)]
SVM_1_df = SVM_1_df.reset_index(drop=True)
SVM_2_df = SVM_2_df.reset_index(drop=True)

#----------------------------------
#----------------------------------
# Define the first SVM
#----------------------------------
#----------------------------------

svm1_inputs = [	'x3',
				'cys1_ss_array0',
				'cys1_ss_array1',
				'cys1_ss_array2',
				'cys1_Ha',
				'cys1_Ca',
				'cys1_Cb',
				'cys1_before Ha',
				'cys1_before Ca']

SVM_1_df   = SVM_1_df[svm1_inputs]

#----------------------------------
#----------------------------------
#  DEFINE SVM-1 DF
#  Define X3 as the training target:
# 		(0 == = -90 and 1 == +90)
#----------------------------------
#----------------------------------

training_df     = SVM_1_df
training_df     = training_df.reset_index(drop=True)
training_target = training_df['x3']
training_df     = training_df.astype(float)
training_df     = training_df.drop('x3',1)

#----------------------------------
#----------------------------------
# SVM_1_df uses MinMax Scaler
# Save SVM and Scaler
#----------------------------------
#----------------------------------
scaler          = preprocessing.MinMaxScaler().fit(training_df)
training_scaled = scaler.transform(training_df)
clf_svm1             = svm.SVC(kernel='rbf',gamma=0.4,C=20,probability=True)
clf_svm1.fit(training_scaled,training_target)


joblib.dump(clf_svm1, 'SVM_1.pkl')
scalerfile_svm1      = 'scaler_SVM_1.sav'
pickle.dump(scaler, open(scalerfile_svm1, 'wb'))


#----------------------------------
#----------------------------------
# Define the second SVM
#----------------------------------
#----------------------------------

svm2_inputs = [
				'x3',
				'x1_array0',
				'x1_array1',
				'x1_array2',
				'cys1_Ha',
				'cys1_Ca',
				'cys1_Cb']

SVM_2_df   =  SVM_2_df[svm2_inputs]

#----------------------------------
#----------------------------------
#  DEFINE SVM-2 DF
#  Define X3 as the training target:
# 		(0 == = -90 and 1 == +90)
#----------------------------------
#----------------------------------


training_df     = SVM_2_df
training_df     = training_df.reset_index(drop=True)
training_target = training_df['x3']
training_df     = training_df.astype(float)
training_df     = training_df.drop('x3',1)


#----------------------------------
#----------------------------------
# SVM_1_df uses StandardScaler
# Save SVM and Scaler
#----------------------------------
#----------------------------------

scaler = preprocessing.StandardScaler().fit(training_df)
training_scaled =scaler.transform(training_df)
clf_svm2=svm.SVC(kernel='rbf',gamma=0.5,C=1,probability=True)
clf_svm2.fit(training_scaled,training_target)
joblib.dump(clf_svm2, 'SVM_2.pkl')
scalerfile_svm2 = 'scaler_SVM_2.sav'
pickle.dump(scaler, open(scalerfile_svm2, 'wb'))
#
