import sys
from sklearn.metrics import *
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from random import shuffle
import numpy as np
import pandas as pd
from pandas import Series
import pickle
from sklearn.externals import joblib
#------------------------------------------------------------------------
#  Load the SVM files
#-----------------------------------------------------------------------
SVM_1 = joblib.load('./SVM_x3/SVM_1.pkl')
SVM_2 = joblib.load('./SVM_x3/SVM_2.pkl')

#------------------------------------------------------------------------
# Load the Saved Scaler for minus60 SVM
#------------------------------------------------------------------------
scalerfile_SVM_1 = './SVM_x3/scaler_SVM_1.sav'
scaler_SVM_1     = pickle.load(open(scalerfile_SVM_1, 'rb'))

scalerfile_SVM_2 = './SVM_x3/scaler_SVM_2.sav'
scaler_SVM_2     = pickle.load(open(scalerfile_SVM_2, 'rb'))

#----------------------------------
# Using the SVMs generated by optimisation on the talox-x3 database, predict the x3 angles for the expanded database
#----------------------------------
pred_dict    = {}
pred_x3_dict = {}



#----------------------------------
# Define split_columns to be used to divide the secondary struture and x1 hot_arrays
#----------------------------------
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


def crosslink_x3_prediction(testing_df):
	
	#----------------------------------
	# Convert X1 angle into array
	#----------------------------------
	
	def x1_array(x1):
		configuration                         = [60,-60,180]
		x1_hot_array                          = [0. for _ in range(3)]
		x1_hot_array[configuration.index(x1)] = 1
		x1_hot_array                          = [str(x) for x in x1_hot_array]
		x1_hot_array                          = ",".join(x1_hot_array)
		return (x1_hot_array)
	
	testing_df['x1_array']  = testing_df['x1'].apply(x1_array)
	
	
	#----------------------------------
	# Divide the dataframe based on x1 angle
	# The x1 angle determines the SVM used and therefore inputs required
	#----------------------------------
	testing_1_df  = testing_df.loc[testing_df['x1'] == (-60)]
	if len(testing_1_df) != 0:
		testing_1_df = testing_1_df.reset_index(drop=True)
		testing_1_df = split_columns(testing_1_df,'ss_array')
		testing_1_df = testing_1_df[['PDB',
						'residue',
						'ss_array0',
						'ss_array1',
						'ss_array2',
						'HA',
						'CA',
						'CB',
						'before_HA',
						'before_CA']]
	
	
	
	
	
	testing_2_df  = testing_df.loc[testing_df['x1'] != (-60)]
	if len(testing_2_df) != 0:
		testing_2_df  = testing_2_df.reset_index(drop=True)
		testing_2_df  = split_columns(testing_2_df,'x1_array')
		testing_2_df  =  testing_2_df[['PDB',
						'residue',
						'x1_array0',
						'x1_array1',
						'x1_array2',
						'HA',
						'CA',
						'CB']]
	
	#----------------------------------
	# Define training for SVM-1
	#----------------------------------
	for index, row in testing_1_df.iterrows():
		testing_inputs_1 = row.drop('PDB')
		testing_inputs_1 = testing_inputs_1.drop('residue')
		testing_inputs_1 = np.array(testing_inputs_1).reshape((1, -1))
		testing_inputs_1 = scaler_SVM_1.transform(testing_inputs_1)
		predict          =  SVM_1.predict(testing_inputs_1)[0]
		x3_hot_array     = [0. for _ in range(2)]
		if predict == -90.0:
			x3_hot_array[1] = 1
		if predict == 90.0:
			x3_hot_array[0] = 1
		x3_hot_array = [str(x) for x in x3_hot_array]
	
		x3_hot_array                 = ",".join(x3_hot_array)
		pred_dict[row['residue'   ]] = x3_hot_array
		pred_x3_dict[row['residue']] = predict
	
	#----------------------------------
	# Define training for SVM-1
	#----------------------------------
	for index, row in testing_2_df.iterrows():
		testing_inputs_2 = row.drop('PDB')
		testing_inputs_2 = testing_inputs_2.drop('residue')
		testing_inputs_2 = np.array(testing_inputs_2).reshape((1, -1))
		testing_inputs_2 = scaler_SVM_2.transform(testing_inputs_2)
		predict          =  SVM_2.predict(testing_inputs_2)[0]
		
		x3_hot_array = [0. for _ in range(2)]
		if predict == -90.0:
			x3_hot_array[1] =1
		if predict == 90.0:
			x3_hot_array[0] =1
		x3_hot_array                 = [str(x) for x in x3_hot_array]
		x3_hot_array                 = ",".join(x3_hot_array)
		pred_dict[row['residue'   ]] = x3_hot_array
		pred_x3_dict[row['residue']] = predict
	
		
	for index, row in testing_df.iterrows():
		predict_array                    = pred_dict[row['residue']]	
		predict_x3                       = pred_x3_dict[row['residue']]	
		testing_df.loc[index,'x3_array'] = predict_array
		testing_df.loc[index,'x3'      ] = predict_x3
	
	return (testing_df)

