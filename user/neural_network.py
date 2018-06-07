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
from sklearn.utils import class_weight
from sklearn.preprocessing import Imputer
from keras.layers import Dropout


###############################################################################################
###############################################################################################
#-----------------------------------------------------------------------
# Keras does not have a matthews correlation coefficient function in metrics to assess the accuracy
# However it does allow you to add a custom metrics
# Therefore will define own MCC function below 
# FROM https://github.com/GeekLiB/keras/blob/master/keras/metrics.py
#-----------------------------------------------------------------------
###############################################################################################
###############################################################################################


import keras.backend as K
def matthews_correlation(y_true, y_pred):
    y_pred_pos  = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg  = 1 - y_pred_pos    
    y_pos       = K.round(K.clip(y_true, 0, 1))
    y_neg       = 1 - y_pos
    tp          = K.sum(y_pos * y_pred_pos)
    tn          = K.sum(y_neg * y_pred_neg)
    fp          = K.sum(y_neg * y_pred_pos)
    fn          = K.sum(y_pos * y_pred_neg)
    numerator   = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())

###############################################################################################
###############################################################################################
#----------------------------------
# Define split column function
#----------------------------------
###############################################################################################
###############################################################################################


def split_columns(dataframe,hot_array):
    length  = dataframe[hot_array][0]
    length  = len(length.split(','))
    s       = dataframe[hot_array].str.split('').apply(Series, 1).stack ()
    s.index = s.index.droplevel(-1)
    s.name  = hot_array
    del dataframe[hot_array]
    dataframe = dataframe.join(s.apply(lambda x: Series(x.split(','))))
    for i in range(length):
         x = str(hot_array)+str(i)
         dataframe = dataframe.rename(columns = {i:x})
    return dataframe


###############################################################################################
###############################################################################################
#----------------------------------
# Keras Prediction
#----------------------------------
###############################################################################################
###############################################################################################

def keras_prediction(training_ensemble, testing_inputs):
        training_ensemble=training_ensemble.reset_index(drop=True)

        #----------------------------------
        # DEFINE THE TRAINING TARGET. IS THE CONNECTIVITY ARRAY
        # DROP FROM THE TRAINING INPUTS
        #----------------------------------
        training_target         = training_ensemble[['connectivity_array']]
        training_target         = split_columns(training_target,'connectivity_array')
        training_inputs         = training_ensemble.drop('connectivity_array',1)
        training_inputs         = training_inputs.astype(float)
        training_target         = training_target.astype(float) 
        
        

        training_inputs_scaled = training_inputs.copy(deep = True)
        testing_inputs_scaled  = testing_inputs.copy(deep = True)
        # TEST SCALE
        column_names = list(testing_inputs)
        for name in column_names:
            meanx = training_inputs[name].mean()
            stdx                                = training_inputs[name].std()
            training_inputs_scaled[name      ] -= meanx
            training_inputs_scaled[name      ] /= stdx
            testing_inputs_scaled[name       ] -= meanx
            testing_inputs_scaled[name       ] /= stdx
            print testing_inputs_scaled[name ][0]
        
        training_inputs_scaled = training_inputs_scaled.replace(np.nan,0)
        training_inputs_scaled = training_inputs_scaled.as_matrix()
        testing_inputs_scaled  = testing_inputs_scaled.replace(np.nan,0)
        testing_inputs_scaled  = testing_inputs_scaled.as_matrix()



        ###############################################################################################
        ###############################################################################################
        #----------------------------------
        # Start the model
        #----------------------------------
        ###############################################################################################
        ###############################################################################################
        
        # Convert the inputs and targets from dataframe to array (required for Keras / TF) 
        training_target        = training_target.as_matrix()
        
        ###############################################################################################
        ###############################################################################################
        #--------------------------------------------------------------------
        # START TO DEFINE THE NEURAL NETWORK
        #--------------------------------------------------------------------
        ###############################################################################################
        ###############################################################################################

        epoch_number           = 200
        batch_size_number      = 200
        input_length           = len(training_inputs_scaled[0])
        opz 				   = keras.optimizers.Adam() 
        

        ###############################################################################################
        ###############################################################################################      
        # Use class weight based on sklearn 'balanced' module
        #----------------------------------
        ###############################################################################################
        ###############################################################################################

        y_index = []
        for _ in training_target:
            _ = _.tolist()
            y_index.append(_.index(1))
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_index), y_index)
    
        ###############################################################################################
        ###############################################################################################
        #----------------------------------
        # Network architechture 
        #----------------------------------
        ###############################################################################################
        ###############################################################################################

        model                              = Sequential()
        model.add(Dropout(0.1, input_shape = (input_length,)))
        model.add(Dense  (100,  input_dim  = input_length, init   = 'normal', activation = 'relu'))
        model.add(Dropout(0.1, input_shape = (input_length,)))
        model.add(Dense  (100,  input_dim  = input_length, init   = 'normal', activation = 'relu'))
        model.add(Dropout(0.1, input_shape = (input_length,)))
        model.add(Dense  (100,  input_dim  = input_length, init   = 'normal', activation = 'relu'))
        model.add(Dense  (02,  init        = 'normal', activation = 'softmax'))
     
        
        ###############################################################################################
        ###############################################################################################
        #----------------------------------
        # Compile the model
        #----------------------------------
        ###############################################################################################
        ###############################################################################################
        model.compile ( loss = 'categorical_crossentropy', optimizer = opz, metrics = ['accuracy', matthews_correlation])
        
        
        history = model.fit( training_inputs_scaled, training_target, epochs =     epoch_number,     batch_size = batch_size_number, class_weight = class_weights, verbose = False )
    	
    	
    
        prediction = model.predict(testing_inputs_scaled)
        predx      = [round(x[1]) for x in prediction]
        max_pred = []
    
        for _ in prediction:
            max_pred.append(max(_))
    
        return (predx,max_pred)