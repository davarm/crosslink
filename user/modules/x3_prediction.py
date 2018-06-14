import sys
from sklearn import svm
from sklearn import preprocessing
from random import shuffle
import numpy as np
import pandas as pd
from pandas import Series
import pickle
from sklearn.externals import joblib

######################################
#------------------------------------
#  Load the SVM files
#------------------------------------
######################################

SVM_1 = joblib.load('./SVM_x3/SVM_1.pkl')
SVM_2 = joblib.load('./SVM_x3/SVM_2.pkl')

######################################
#------------------------------------
# Load the scaler files
#------------------------------------
######################################

scalerfile_SVM_1 = './SVM_x3/scaler_SVM_1.sav'
scaler_SVM_1     = pickle.load(open(scalerfile_SVM_1, 'rb'))

scalerfile_SVM_2 = './SVM_x3/scaler_SVM_2.sav'
scaler_SVM_2     = pickle.load(open(scalerfile_SVM_2, 'rb'))

######################################
#------------------------------------
# Store x3 prediction (array and value) in two dictionaries
#------------------------------------
######################################
pred_dict    = {}
pred_x3_dict = {}

######################################
#------------------------------------
# Define split_columns to be used to divide the secondary struture and x1 hot_arrays
#------------------------------------
######################################

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

######################################
#------------------------------------
# Define function to be called as the first stage
# of connectivity_prediction.py
#------------------------------------
######################################

def crosslink_x3_prediction(testing_df):
	
	######################################
	#------------------------------------
	# Convert X1 angle into array
	#------------------------------------
	#####################################
	
	def x1_array(x1):
		configuration                         = [60,-60,180]
		x1_hot_array                          = [0. for _ in range(3)]
		x1_hot_array[configuration.index(x1)] = 1
		x1_hot_array                          = [str(x) for x in x1_hot_array]
		x1_hot_array                          = ",".join(x1_hot_array)
		return (x1_hot_array)
	
	testing_df['x1_array']  = testing_df['x1'].apply(x1_array)
	
	
	######################################
	#------------------------------------
	# Divide the dataframe based on x1 angle
	# The x1 angle determines the SVM used and therefore inputs required
	# SVM-1 if X1 == -60
	#------------------------------------
	#####################################

	testing_1_df  = testing_df.loc[testing_df['x1'] == (-60)]
	if len(testing_1_df) != 0:
		testing_1_df = testing_1_df.reset_index(drop=True)
		testing_1_df = split_columns(testing_1_df,'ss_array')

		######################################
		#------------------------------------
		# WARNING
		# THE ORDER OF INPUTS MUST BE THE SAME AS THOSE USED TO TRAIN THE SVM
		# THIS IS FOUND IN THE SVM_X3 DIRECTORY UNDER "generating_SVM.py"
		#------------------------------------
		#####################################
		testing_1_df = testing_1_df[[
						'PDB',
						'residue',
						'ss_array0',
						'ss_array1',
						'ss_array2',
						'HA',
						'CA',
						'CB',
						'before_HA',
						'before_CA']]
	
	
	######################################
	#------------------------------------
	# Divide the dataframe based on x1 angle
	# The x1 angle determines the SVM used and therefore inputs required
	# SVM-2 if X2 == +60 or 180
	#------------------------------------
	#####################################	
	
	
	testing_2_df  = testing_df.loc[testing_df['x1'] != (-60)]
	if len(testing_2_df) != 0:
		testing_2_df  = testing_2_df.reset_index(drop=True)
		testing_2_df  = split_columns(testing_2_df,'x1_array')

		######################################
		#------------------------------------
		# WARNING
		# THE ORDER OF INPUTS MUST BE THE SAME AS THOSE USED TO TRAIN THE SVM
		# THIS IS FOUND IN THE SVM_X3 DIRECTORY UNDER "generating_SVM.py"
		#------------------------------------
		#####################################
		testing_2_df  =  testing_2_df[[
						'PDB',
						'residue',
						'x1_array0',
						'x1_array1',
						'x1_array2',
						'HA',
						'CA',
						'CB']]
	
	######################################
	#------------------------------------
	# START PREDICTION FOR TESTING_1_DF
	# Iterate through each row individually so that results can be 
	# stored in a dictionary
	#------------------------------------
	#####################################

	for index, row in testing_1_df.iterrows():
		testing_inputs_1 = row.drop('PDB')
		testing_inputs_1 = testing_inputs_1.drop('residue')
		testing_inputs_1 = np.array(testing_inputs_1).reshape((1, -1))

		######################################
		#------------------------------------
		# Scale the data with the previously saved scaler, based on the training data
		# For SVM_1 a MinMaxScaler was used
		#------------------------------------
		#####################################

		testing_inputs_1 = scaler_SVM_1.transform(testing_inputs_1)
		
		######################################
		#------------------------------------
		# Return prediction and store in dictionary with residue as key
		# Also generate a hot_array to include as inputs (required for connectivity)
		#   	+90 = [1,0]
		#		-90 = [0,1]
		#------------------------------------
		#####################################
		predict = SVM_1.predict(testing_inputs_1)[0]

		x3_hot_array = [0. for _ in range(2)]
		if predict == -90.0:
			x3_hot_array[1] = 1
		if predict == 90.0:
			x3_hot_array[0] = 1
		x3_hot_array = [str(x) for x in x3_hot_array]
	
		x3_hot_array                 = ",".join(x3_hot_array)
		pred_dict[row['residue'   ]] = x3_hot_array
		pred_x3_dict[row['residue']] = predict
	
	######################################
	#------------------------------------
	# START PREDICTION FOR TESTING_2_DF
	# Iterate through each row individually so that results can be 
	# stored in a dictionary
	#------------------------------------
	#####################################
	for index, row in testing_2_df.iterrows():
		testing_inputs_2 = row.drop('PDB')
		testing_inputs_2 = testing_inputs_2.drop('residue')
		testing_inputs_2 = np.array(testing_inputs_2).reshape((1, -1))

		######################################
		#------------------------------------
		# Scale the data with the previously saved scaler, based on the training data
		# For SVM_1 a StandardScaler was used
		#------------------------------------
		#####################################
		testing_inputs_2 = scaler_SVM_2.transform(testing_inputs_2)

		######################################
		#------------------------------------
		# Return prediction and store in dictionary with residue as key
		# Also generate a hot_array to include as inputs (required for connectivity)
		#   	+90 = [1,0]
		#		-90 = [0,1]
		#------------------------------------
		#####################################
		predict = SVM_2.predict(testing_inputs_2)[0]
		
		x3_hot_array = [0. for _ in range(2)]
		if predict == -90.0:
			x3_hot_array[1] =1
		if predict == 90.0:
			x3_hot_array[0] =1
		x3_hot_array                 = [str(x) for x in x3_hot_array]
		x3_hot_array                 = ",".join(x3_hot_array)
		pred_dict[row['residue'   ]] = x3_hot_array
		pred_x3_dict[row['residue']] = predict
	
	
	######################################
	#------------------------------------
	# Iterate through the original dataframe and add in the predicted values
	#------------------------------------
	#####################################

	def add_x3(residue):
		predicted_x3 = pred_x3_dict[residue]
		return(predicted_x3)
	testing_df['x3'] = testing_df['residue'].apply(add_x3)

	def add_x3_array(residue):
		predicted_x3 = pred_dict[residue]
		return(predicted_x3)
	testing_df['x3_array'] = testing_df['residue'].apply(add_x3_array)
	
	return (testing_df)

