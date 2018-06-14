import numpy as np
from sklearn.metrics import *
from random import shuffle
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

path = "./neural_network/"
training_inputs = pd.read_csv('./modules/training_database.csv')


#########################################################################################################
# Function returns the prediction 0 == Not Connected, 1 == Connected
# and the prrobability of the prediction
#########################################################################################################

def keras_prediction(testing_inputs,i):
    testing_inputs_scaled = testing_inputs.copy(deep = True)
    column_names          = list(testing_inputs_scaled)

    #########################################################################################################
    # Scale the data by substracting the mean and then dividing by the standard deviation for each column
    # Mean and standard deviation based on the training_database values
    # Scaling based on as per Francois Chollet: Deep Learning with Python (2017). Manning Publications
    #####################################################################################################
    for name in column_names:
        meanx = training_inputs[name].mean()
        stdx                                = training_inputs[name].std()
        testing_inputs_scaled[name       ] -= meanx
        testing_inputs_scaled[name       ] /= stdx

    #########################################################################################################
    # Do not scale missing nuclei chemical shift, keep them as NaN and then replace as zero afte scaling
    ###########################################################################################################
    testing_inputs_scaled = testing_inputs_scaled.replace(np.nan,0)
    testing_inputs_scaled = testing_inputs_scaled.as_matrix()
    
    #########################################################################################################
    # Read the saved models in neural network directory and then compile
    #########################################################################################################
    model                  = load_model(path+str(i)+"_model.h5")
    model.compile(loss     ='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    prediction             = model.predict(testing_inputs_scaled)
    predx                  = [round(x[1]) for x in prediction]
    probability_prediction = []
    print prediction
    
    for _ in prediction:
        probability_prediction.append(max(_))
    #########################################################################################################
    # Return the prediction and the probability 
    #########################################################################################################
    return (predx,probability_prediction)