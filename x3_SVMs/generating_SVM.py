import sys
# sys.path.append('../data_proc/database_generation_x3')
# from database_generation_x3_prediction import database_generation
#import shuffle
from sklearn.metrics import *
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from random import shuffle
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.externals import joblib
import pickle


df         = pd.read_csv('Cys_database.csv',sep = ',', skipinitialspace = False)

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
SVM_1_df = df.loc[df['x1'] == (-60)]
SVM_2_df   = df.loc[df['x1'] != (-60)]
SVM_1_df = SVM_1_df.reset_index(drop=True)
SVM_2_df   = SVM_2_df.reset_index(drop=True)

SVM_1_df   = SVM_1_df[[	'x3',
					'cys1_ss_array0',
					'cys1_ss_array1',
					'cys1_ss_array2',
					'cys1_Ha',
					'cys1_Ca',
					'cys1_Cb',
					'cys1_before Ha',
					'cys1_before Ca']]

SVM_2_df   =  SVM_2_df[[
					'x3',
					'x1_array0',
					'x1_array1',
					'x1_array2',
					'cys1_Ha',
					'cys1_Ca',
					'cys1_Cb']]


print len(SVM_1_df)



#------------------------------------------------------------------------
#  DEFINE SVM-1 DF
# ------------------------------------------------------------------------
training_df     = SVM_1_df
training_df     = training_df.reset_index(drop=True)
training_target = training_df['x3']
# training_df     = training_df.drop('PDB',1)
training_df     = training_df.astype(float)
training_df     = training_df.drop('x3',1)

#
#
scaler = preprocessing.MinMaxScaler().fit(training_df)
training_scaled =scaler.transform(training_df)
clf=svm.SVC(kernel='rbf',gamma=0.4,C=20,probability=True)
clf.fit(training_scaled,training_target)
joblib.dump(clf, 'SVM_1.pkl')
#
scalerfile = 'scaler_SVM_1.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))
#------------------------------------------------------------------------
# Writing SVM-2
###------------------------------------------------------------------------
#training_df     = SVM_2_df
#training_df     = training_df.reset_index(drop=True)
#training_target = training_df['x3']
## training_df     = training_df.drop('PDB',1)
#training_df     = training_df.astype(float)
#training_df     = training_df.drop('x3',1)
#
#
#
#scaler = preprocessing.StandardScaler().fit(training_df)
#training_scaled =scaler.transform(training_df)
#clf=svm.SVC(kernel='rbf',gamma=0.5,C=1,probability=True)
#clf.fit(training_scaled,training_target)
#joblib.dump(clf, 'SVM_2.pkl')
#
#scalerfile = 'scaler_SVM_2.sav'
#pickle.dump(scaler, open(scalerfile, 'wb'))
#
