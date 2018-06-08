from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import *
from random import shuffle
import pandas as pd
from pandas import Series
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.layers import Dropout
import math
from sklearn.utils import class_weight
from random import shuffle
from neural_network import keras_prediction
from collections import Counter

########################################################################
########################################################################
#--------------------------------------------------------------------------
# A bagging neural network for prediction of disulfide connectivity
# Testing on training dataset by 10 * k cross fold
# Do this by taking out 10% of the PDBs for validation, test by bagging then take the next 10%
# START BY DEFINING REQUIRED FUNCTIONS AND INPUTS
#--------------------------------------------------------------------------
########################################################################
########################################################################



########################################################################
########################################################################
#--------------------------------------------------------------------------
# Split columnt function
#--------------------------------------------------------------------------
########################################################################
########################################################################

def split_columns(dataframe,hot_array):
    # print dataframe[hot_array][0]
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





        
########################################################################
########################################################################
#--------------------------------------------------------------------------
# Desired inputs used for training and testing
#--------------------------------------------------------------------------
########################################################################
########################################################################

inputs = [  
            'cys1_Ha',
            'cys2_Ha',
            'cys1_N',
            'cys2_N',
            'cys1_Hn',
            'cys2_Hn',
            'cys1_before Hn',
            'cys2_before Hn',
            'cys1_after Hn',
            'cys2_after Hn',
            'cys1_before Ca',
            'cys2_before Ca',
            'cys1_after Ca',
            'cys2_after Ca',
            'cys1_before Ha',
            'cys2_before Ha',
            'cys1_after Ha',
            'cys2_after Ha',
            'cys1_ss_array',
            'cys2_ss_array',
            'cys1_x1_array',
            'cys2_x1_array',
            'cys1_x3_array',
            'cys2_x3_array',
            'cys_diff',
            'no_disulfides',
            'connectivity_array']

nuclei_list = [  
            'cys1_Ha',
            'cys2_Ha',
            'cys1_N',
            'cys2_N',
            'cys1_Hn',
            'cys2_Hn',
            'cys1_before Hn',
            'cys2_before Hn',
            'cys1_after Hn',
            'cys2_after Hn',
            'cys1_before Ca',
            'cys2_before Ca',
            'cys1_after Ca',
            'cys2_after Ca',
            'cys1_before Ha',
            'cys2_before Ha',
            'cys1_after Ha',
            'cys2_after Ha']

testing_inputs_list = [  
            'cys1_Ha',
            'cys2_Ha',
            'cys1_N',
            'cys2_N',
            'cys1_Hn',
            'cys2_Hn',
            'cys1_before Hn',
            'cys2_before Hn',
            'cys1_after Hn',
            'cys2_after Hn',
            'cys1_before Ca',
            'cys2_before Ca',
            'cys1_after Ca',
            'cys2_after Ca',
            'cys1_before Ha',
            'cys2_before Ha',
            'cys1_after Ha',
            'cys2_after Ha',
            'cys1_ss_array',
            'cys2_ss_array',
            'cys1_x1_array',
            'cys2_x1_array',
            'cys1_x3_array',
            'cys2_x3_array',
            'cys_diff',
            'no_disulfides']
    
########################################################################
########################################################################
#--------------------------------------------------------------------------
# Store all individual PDBS in a list
#--------------------------------------------------------------------------
########################################################################
########################################################################



########################################################################
########################################################################
#--------------------------------------------------------------------------
# Read the connectivity database into a pandas dataframe
#--------------------------------------------------------------------------
########################################################################
########################################################################


training_df = pd.read_csv('training_database.csv', sep = ',', skipinitialspace = False)


#----------------------------------
# The Desired Inputs to use for training
#----------------------------------
training_df = training_df.sample(frac=1).reset_index(drop=True)




########################################################################
########################################################################
#----------------------------------
# replace nuclei zeros with nan
#----------------------------------
########################################################################

for nuclei in nuclei_list:
  training_df[nuclei] = training_df[nuclei].replace(0,np.nan)
training_df = training_df.reset_index(drop = True ) 





testing_df_og              = pd.read_csv('all.csv', sep = ',', skipinitialspace = False)
testing_df_og['cys_diff']  = testing_df_og['cys1_residue_number'] - testing_df_og['cys2_residue_number']    
testing_df_og              = testing_df_og.reset_index(drop=True)
testing_df              = testing_df_og[testing_inputs_list]


########################################################################
########################################################################
#----------------------------------
# replace nuclei zeros with nan
#----------------------------------
########################################################################
for nuclei in nuclei_list:
  testing_df[nuclei] = testing_df[nuclei].replace(0,np.nan)



########################################################################
########################################################################
#----------------------------------
# Split the columns of the testing dataframe
#----------------------------------
########################################################################
########################################################################


#########################
##### WARNING ###########
#########################
# Must split them in the same order as the training dataframe

testing_df             = split_columns(testing_df,'cys1_ss_array')
testing_df             = split_columns(testing_df,'cys2_ss_array')
testing_df             = split_columns(testing_df,'cys1_x1_array')
testing_df             = split_columns(testing_df,'cys2_x1_array')
testing_df             = split_columns(testing_df,'cys1_x3_array')
testing_df             = split_columns(testing_df,'cys2_x3_array')

########################################################################
########################################################################
#----------------------------------
# Define the testing target and inputs
#----------------------------------
########################################################################
########################################################################


testing_inputs         = testing_df
testing_inputs         = testing_inputs.astype(float)


########################################################################
########################################################################
#----------------------------------
# Create dictionary to store results for each bagging ensemble
#----------------------------------
########################################################################
########################################################################

results      = {}
results_prob = {}
for value in range(len(testing_df)):
	results[str(value)     ] = []
	results_prob[str(value)] = []


########################################################################
########################################################################
#----------------------------------
# START THE BAGGING PROCESS
# TAKE 80% of the training_df, train and predict. 
# Then increase by 10%, take 10%-90% and train.... 30%-100% + 0-10%
# The ensemble function defined at the start does this 
#----------------------------------
########################################################################
########################################################################
fract =  (float(len(training_df)) * 0.9)
fract = round(int(fract))
start = 0
s_fract = (float(len(training_df)) * 0.05)
s_fract = round(int(s_fract))


########################################################################
########################################################################
#--------------------------------------------------------------------------
# Ensemble function. To take different ensembles for bagging
#--------------------------------------------------------------------------
########################################################################
########################################################################



def ensemble(training_dfx, start,stop ):
	print 'START, STOP',start, stop   
	if stop <= len(training_dfx):
		ensemble = training_dfx[int(start):int(stop)]    

	if stop > len(training_dfx) and start < len(training_dfx):
		ensemble1 = training_dfx[int(start):] 
		ensemble2 = training_dfx[:int(stop)-len(training_dfx)]
		ensemble = pd.concat([ensemble1, ensemble2])
	return (ensemble)



columns =   ['PDB',
            'cys1',
            'cys2']


results_df = pd.DataFrame(index = range(len(testing_df_og)),columns = columns)
results_df['PDB'] =  testing_df_og['PDB']
results_df['cys1'] = testing_df_og['cys1_residue_number']
results_df['cys2'] = testing_df_og['cys2_residue_number']

i = 0
while i <9:
  print 'TRAINING ENSMEBLE',i 
  print 'start',start,'stop',fract
  ensemble_df    = ensemble(training_df, start,fract)

  ########################################################################
  ########################################################################
  #----------------------------------
  # Use keras_prediciton from the neural_network module that defines the nn for predction
  #----------------------------------
  ########################################################################
  #######################################################################
  prediction     = keras_prediction(ensemble_df, testing_inputs)
  max_prediction = prediction[1]
  prediction     = prediction[0]
   
  ########################################################################
  ########################################################################
  #----------------------------------
  # Store results in prediction in dictionaries previously defined
  # Have just the prediction and the probability of prediciton
  #----------------------------------
  ########################################################################
  #######################################################################
  results_df['prediction'+str(i)] = prediction
  results_df['probability'+str(i)] = max_prediction
  for k, pred in enumerate(prediction):
  	results[str(k)].append(pred)
  	results_prob[str(k)].append(max_prediction[k])
  	



  fract = fract + s_fract
  start = start + s_fract        
  i = i+1


prediction_df = results_df[['prediction0',
'prediction1',
'prediction2',
'prediction3',
'prediction4',
'prediction5',
'prediction6',
'prediction7',
'prediction8']]

probability_df = results_df[['probability0',
'probability1',
'probability2',
'probability3',
'probability4',
'probability5',
'probability6',
'probability7',
'probability8']]

########################################################################
########################################################################
#----------------------------------
# Start to analyse the resuls
# Most common function will return the most common prediciton out of the 10 predictions
#----------------------------------
########################################################################
#######################################################################

from collections import Counter
def Most_Common(lst):
  data = Counter(lst)
  return data.most_common(1)[0][0]


for index,row in prediction_df.iterrows():
    row = row.tolist()
    prediction =(Most_Common(row))    
    results_df.loc[index,'prediction'] = prediction

results_df.to_csv('results.csv',index = False)
for index,row in probability_df.iterrows():

    row = row.tolist()
    probability = np.mean(row) 
    results_df.loc[index,'probability'] = probability
    
results_df = results_df[['PDB','cys1','cys2','prediction','probability']]
for index,row in results_df.iterrows():
   print row.tolist()
