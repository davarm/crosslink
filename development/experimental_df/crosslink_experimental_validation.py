from numpy.random import seed
seed(1)
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import *
from random import shuffle
import pandas as pd
from pandas import Series
import math
from sklearn.utils import class_weight
from random import shuffle
from neural_network import keras_prediction
from collections import Counter


########################################################################
########################################################################
#--------------------------------------------------------------------------
# Testing the neural network on the experimental databases
#--------------------------------------------------------------------------
########################################################################
########################################################################



########################################################################
########################################################################
#--------------------------------------------------------------------------
# Split column function
#--------------------------------------------------------------------------
########################################################################
########################################################################

def split_columns(dataframe,hot_array):
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


  
########################################################################
########################################################################
#--------------------------------------------------------------------------
# Read the connectivity database into a pandas dataframe
# To be used as the training database
#--------------------------------------------------------------------------
########################################################################
########################################################################
training_df_init = pd.read_csv('peptide_df_connectivity.csv', sep = ',', skipinitialspace = False)

#----------------------------------
# The Desired Inputs to use for training
#----------------------------------
training_df_init  = training_df_init[inputs]



########################################################################
########################################################################
#--------------------------------------------------------------------------
# FOR TRAINING DF, to generate extra inputs we double the databse by 'reversing' the orders of cysteine residues
# E.g. Cys1, Cys2 to Cys2, Cys1
#--------------------------------------------------------------------------
########################################################################
########################################################################

training_df_reverse = training_df_init[inputs]

#########################B###############################################
########################################################################
#--------------------------------------------------------------------------
# Rename the columns, changing Cys1 to Cys2 and Cys2 to Cys1
#--------------------------------------------------------------------------
########################################################################
########################################################################

training_df_reverse = training_df_reverse.rename(columns={
                                      'cys1_Ha'                   : 'cys2_Ha', 
                                      'cys2_Ha'                   : 'cys1_Ha', 
                                      'cys1_N'                    : 'cys2_N', 
                                      'cys2_N'                    : 'cys1_N', 
                                      'cys1_Hn'                   : 'cys2_Hn', 
                                      'cys2_Hn'                   : 'cys1_Hn', 
                                      'cys1_psi'                  : 'cys2_psi', 
                                      'cys2_psi'                  : 'cys1_psi', 
                                      'cys1_before Hn'            : 'cys2_before Hn', 
                                      'cys2_before Hn'            : 'cys1_before Hn', 
                                      'cys1_after Hn'             : 'cys2_after Hn', 
                                      'cys2_after Hn'             : 'cys1_after Hn', 
                                      'cys1_before Ca'            : 'cys2_before Ca', 
                                      'cys2_before Ca'            : 'cys1_before Ca', 
                                      'cys1_after Ca'             : 'cys2_after Ca', 
                                      'cys2_after Ca'             : 'cys1_after Ca', 
                                      'cys1_before Ha'            : 'cys2_before Ha', 
                                      'cys2_before Ha'            : 'cys1_before Ha', 
                                      'cys1_after Ha'             : 'cys2_after Ha', 
                                      'cys2_after Ha'             : 'cys1_after Ha', 
                                      'cys1_ss_array'             : 'cys2_ss_array', 
                                      'cys2_ss_array'             : 'cys1_ss_array',  
                                      'cys1_x1_array'             : 'cys2_x1_array', 
                                      'cys2_x1_array'             : 'cys1_x1_array',
                                      'cys1_x3_array'             : 'cys2_x3_array', 
                                      'cys2_x3_array'             : 'cys1_x3_array',
                                      })
  
########################################################################
########################################################################
#--------------------------------------------------------------------------
  # ENSURE INPUTS ARE IN THE SAME ORDER AND THEN JOIN THE TWO DATABAES TOGETHER
#--------------------------------------------------------------------------
########################################################################
########################################################################

training_df_reverse = training_df_reverse[inputs]
training_df         = training_df_init.append(training_df_reverse, ignore_index = True)
print 'TRAINING DF',len(training_df)

########################################################################
########################################################################
#----------------------------------
# REPLACE ALL '0' values with nan
#----------------------------------
########################################################################
########################################################################
for nuclei in nuclei_list:
  training_df[nuclei] = training_df[nuclei].replace(0,np.nan)
training_df = training_df.reset_index(drop = True )
print len(training_df)

training_df = training_df.replace(np.nan,0)
print training_df

########################################################################
########################################################################
#--------------------------------------------------------------------------
  # Split the hot_arrays into individal columns as per the split_columns function 
#--------------------------------------------------------------------------
########################################################################
########################################################################

training_df             = split_columns(training_df,'cys1_ss_array')
training_df             = split_columns(training_df,'cys2_ss_array')    
training_df             = split_columns(training_df,'cys1_x1_array')
training_df             = split_columns(training_df,'cys2_x1_array')
training_df             = split_columns(training_df,'cys1_x3_array')
training_df             = split_columns(training_df,'cys2_x3_array')

########################################################################
########################################################################
#--------------------------------------------------------------------------
# SHUFFLE THE TRAINING DF
#--------------------------------------------------------------------------
########################################################################
########################################################################

training_df = training_df.sample(frac=1).reset_index(drop=True)
  

########################################################################
########################################################################
#--------------------------------------------------------------------
# Define the testing database. Will be based on the experimental database
# where chemical shifts have been experimentally determined
#----------------------------------
########################################################################
########################################################################

testing_df_og = pd.read_csv('experimental_df_connectivity.csv', sep = ',', skipinitialspace = False)
testing_df_og = testing_df_og.replace([np.NaN],['0'])
testing_df_og = testing_df_og.reset_index(drop=True)
testing_df_og = testing_df_og.loc[testing_df_og['cys1_Cb'] != (0)]
testing_df_og = testing_df_og.loc[testing_df_og['cys2_Cb'] != (0)]
testing_df_og = testing_df_og.reset_index(drop=True)
testing_df    = testing_df_og[inputs]

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
# Connectivity Array, 0 = Not Connected (False), 1 = Connected (True)
#----------------------------------
########################################################################
########################################################################

testing_target         = testing_df[['connectivity_array']]
testing_target         = split_columns(testing_target,'connectivity_array')
testing_inputs         = testing_df.drop('connectivity_array',1)
testing_inputs         = testing_inputs.astype(float)
testing_target         = testing_target.astype(float)
testing_target         = testing_target.as_matrix()


########################################################################
########################################################################
#----------------------------------
# Create dictionary to store results for each bagging ensemble
#----------------------------------
########################################################################
########################################################################
truex        = [round(x[1]) for x in testing_target]
results      = {}
results_prob = {}
for value in range(len(truex)):
	results[str(value)     ] = []
	results_prob[str(value)] = []


########################################################################
########################################################################
#----------------------------------
# START THE BAGGING PROCESS
# TAKE 90% of the training_df, train and predict. 
# Then increase by 5%
# The ensemble function defined at the start does this 
#----------------------------------
########################################################################
########################################################################
fract   = (float(len(training_df)) * 0.9)
fract   = round(int(fract))
start   = 0
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
    ensemble  = pd.concat([ensemble1, ensemble2])
	return (ensemble)



columns =   ['PDB',
            'cys1',
            'cys2',
            'actual']


results_df           = pd.DataFrame(index = range(len(testing_df_og)),columns = columns)
results_df['PDB'   ] =  testing_df_og['PDB']
results_df['cys1'  ] = testing_df_og['cys1_residue_number']
results_df['cys2'  ] = testing_df_og['cys2_residue_number']
results_df['actual'] = testing_df_og['connectivity_index']

i = 0
while i <9:
  print 'TRAINING ITERATION',i 
  print 'start',start,'stop',fract
  ensemble_df    = ensemble(training_df, start,fract)
  print 'ENSMEBLE DF'

  ########################################################################
  ########################################################################
  #----------------------------------
  # Use keras_prediciton from the neural_network module that defines the nn for predction
  #----------------------------------
  ########################################################################
  #######################################################################
  prediction     = keras_prediction(ensemble_df, testing_inputs, testing_target)
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

print results_df


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


def Most_Common(lst):
  data = Counter(lst)
  return data.most_common(1)[0][0]


for index,row in prediction_df.iterrows():
    row                                = row.tolist()
    prediction                         = (Most_Common(row))    
    results_df.loc[index,'prediction'] = prediction

for index,row in probability_df.iterrows():

    row = row.tolist()
    probability = np.mean(row) 
    results_df.loc[index,'probability'] = probability

print results_df
print 'Accuracy', accuracy_score(results_df['actual'], results_df['prediction']),'MCC', matthews_corrcoef(results_df['actual'], results_df['prediction'])

results_df_07 = results_df.query('probability > 0.7')
results_df_07 = results_df_07.reset_index(drop = True)

results_df_08 = results_df.query('probability > 0.8')
results_df_08 = results_df_08.reset_index(drop = True)

results_df_09 = results_df.query('probability > 0.9')
results_df_09 = results_df_09.reset_index(drop = True)

print 'Accuracy', accuracy_score(results_df_07['actual'], results_df_07['prediction']),'MCC', matthews_corrcoef(results_df_07['actual'], results_df_07['prediction']), 'Freq',float(len(results_df_07))/float(len(results_df))
print ''
print 'Accuracy', accuracy_score(results_df_08['actual'], results_df_08['prediction']),'MCC', matthews_corrcoef(results_df_08['actual'], results_df_08['prediction']),float(len(results_df_08))/float(len(results_df))
print ''
print 'Accuracy', accuracy_score(results_df_09['actual'], results_df_09['prediction']),'MCC', matthews_corrcoef(results_df_09['actual'], results_df_09['prediction']),float(len(results_df_09))/float(len(results_df))