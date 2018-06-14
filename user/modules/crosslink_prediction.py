import numpy as np
import pandas as pd
from pandas import Series
from crosslink_neural_network import keras_prediction
from collections import Counter

########################################################################
########################################################################
#--------------------------------------------------------------------------
# A bagging neural network for prediction of disulfide connectivity
# Take testing inputs and then call the crosslink neural network
#--------------------------------------------------------------------------
########################################################################
########################################################################



########################################################################
########################################################################
#--------------------------------------------------------------------------
# Desired inputs used for training and testing
#--------------------------------------------------------------------------
########################################################################
########################################################################

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

input_list = [
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
                   'cys_diff',
                   'no_disulfides',
                   'cys1_ss_array0',
                   'cys1_ss_array1',
                   'cys1_ss_array2',
                   'cys2_ss_array0',
                   'cys2_ss_array1',
                   'cys2_ss_array2',
                   'cys1_x1_array0',
                   'cys1_x1_array1',
                   'cys1_x1_array2',
                   'cys2_x1_array0',
                   'cys2_x1_array1',
                   'cys2_x1_array2',
                   'cys1_x3_array0',
                   'cys1_x3_array1',
                   'cys2_x3_array0',
                   'cys2_x3_array1']

def crosslink_prediction(testing_df):
  testing_inputs = testing_df[input_list]
  testing_inputs = testing_inputs.astype(float)
  
  ########################################################################
  ########################################################################
  #----------------------------------
  # Replace unassigned nuclei chemical shifts (0) with NaN
  #----------------------------------
  ########################################################################
  for nuclei in nuclei_list:
    testing_inputs[nuclei] = testing_inputs[nuclei].replace(0,np.nan)
  
  
  ########################################################################
  ########################################################################
  #----------------------------------
  # Create dictionary and dataframe to store results for each bagging ensemble
  #----------------------------------
  ########################################################################
  ########################################################################
  
  results      = {}
  results_prob = {}
  for value in range(len(testing_df)):
  	results[str(value)     ] = []
  	results_prob[str(value)] = []
  
  
  columns =   ['PDB',
              'cys1',
              'cys2']
  
  
  results_df         = pd.DataFrame(index = range(len(testing_df)),columns = columns)
  results_df['PDB' ] =  testing_df['PDB']
  results_df['cys1'] = testing_df['cys1_residue_number']
  results_df['cys2'] = testing_df['cys2_residue_number']
  
  
  ########################################################################
  ########################################################################
  #----------------------------------
  # The neural network uses a 9 * bagging method for the prediciton (therefore)
  # 9 different neural networks.
  # Call each network through the crosslink_neural_network keras_prediction store results 
  #----------------------------------
  ########################################################################
  ########################################################################
  
  def connectivity_prediction(testing_inputs, model_number):
    
    ########################################################################
    ########################################################################
    #----------------------------------
    # Use keras_prediciton from the neural_network module that defines the nn for predction
    #----------------------------------
    ########################################################################
    #######################################################################
    prediction     = keras_prediction(testing_inputs, model_number)
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
    results_df['prediction' +str(model_number)] = prediction
    results_df['probability'+str(model_number)] = max_prediction
  
    for k, pred in enumerate(prediction):
      results[str(k)].append(pred)
      results_prob[str(k)].append(max_prediction[k])
  
    return()
  
  for i in range(0,9):
    connectivity_prediction(testing_inputs, i)
  
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
  # Most common function will return the most common prediciton out of the 9 predictions
  #----------------------------------
  ########################################################################
  #######################################################################
  def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]
  
  
  for index,row in prediction_df.iterrows():
      row        = row.tolist()
      prediction = (Most_Common(row))    
      results_df.loc[index,'prediction'] = prediction
  
  results_df.to_csv('results.csv',index = False)
  for index,row in probability_df.iterrows():
  
      row         = row.tolist()
      probability = np.mean(row) 
      results_df.loc[index,'probability'] = probability
      
  results_df = results_df[['PDB','cys1','cys2','prediction','probability']]
  for index,row in results_df.iterrows():
     print row.tolist()
  return()