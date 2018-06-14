from numpy.random import seed
seed(1)
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
from sklearn.utils import class_weight
from keras.layers import Dropout




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


def keras_prediction(training_ensemble, testing_inputs,testing_target):
        training_ensemble = training_ensemble.reset_index(drop=True)

        ########################################################################
        ########################################################################
        #--------------------------------------------------------------------------
        # DEFINE THE TRAINING TARGET THAT IS REPRESENTED BY THE CONNECTIVITY ARRAY ( 0 = NOT CONNECTED, 1 = CONNECTED)
        # DROP FROM THE TRAINING INPUTS
        #--------------------------------------------------------------------------
        ########################################################################
        ########################################################################

        training_target         = training_ensemble[['connectivity_array']]
        training_target         = split_columns(training_target,'connectivity_array')
        training_inputs         = training_ensemble.drop('connectivity_array',1)
        training_inputs         = training_inputs.astype(float)
        training_target         = training_target.astype(float) 
        
        training_inputs_scaled  = training_inputs.copy(deep = True)
        testing_inputs_scaled   = testing_inputs.copy(deep = True)
        column_names            = list(testing_inputs)

        ########################################################################
        ########################################################################
        #--------------------------------------------------------------------------
        # Scale the inputs by subtracting the mean and dividing by the standard deviation
        #--------------------------------------------------------------------------
        ########################################################################
        ########################################################################

        for name in column_names:
            meanx = training_inputs[name].mean()
            stdx = training_inputs[name].std()
            training_inputs_scaled[name] -= meanx
            training_inputs_scaled[name] /= stdx
            testing_inputs_scaled[name]  -= meanx
            testing_inputs_scaled[name]  /= stdx

        ########################################################################
        ########################################################################
        #--------------------------------------------------------------------------
        # Replace NaN values for nuclei back to 0 to represent missing chemical shifts 
        # (Set as NaNs previously to ensure they are not scaled)
        #--------------------------------------------------------------------------
        ########################################################################
        ########################################################################

        training_inputs_scaled = training_inputs_scaled.replace(np.nan,0)
        training_inputs_scaled = training_inputs_scaled.as_matrix()
        testing_inputs_scaled  = testing_inputs_scaled.replace(np.nan,0)
        testing_inputs_scaled  = testing_inputs_scaled.as_matrix()
        training_target        = training_target.as_matrix()

        ###############################################################################################
        ###############################################################################################
        #----------------------------------
        # Start the model
        #----------------------------------
        ###############################################################################################
        ###############################################################################################
        

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
        #----------------------------------
        # Use class weight based on sklearn 'balanced' module
        #----------------------------------
        ###############################################################################################
        ###############################################################################################

        y_index = []
        for connectivity in training_target:
            connectivity = connectivity.tolist()
            y_index.append(connectivity.index(1))
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
        model.compile ( loss = 'categorical_crossentropy', optimizer = opz, metrics = ['accuracy'])
        history = model.fit( training_inputs_scaled, training_target, validation_data = [testing_inputs_scaled, testing_target], epochs =     epoch_number,     batch_size = batch_size_number, class_weight = class_weights, verbose = False )
    
        prediction = model.predict(testing_inputs_scaled)
        predx      = [round(x[1]) for x in prediction]
        max_pred   = []
    
        for pred in prediction:
            max_pred.append(max(pred))
        return (predx,max_pred)