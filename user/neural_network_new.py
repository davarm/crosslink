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
from keras.models import model_from_json

path = "./neural_network/"

training_inputs = pd.read_csv('training_database.csv')

def keras_prediction(testing_inputs,i):
    testing_inputs_scaled  = testing_inputs.copy(deep = True)
    column_names = list(testing_inputs_scaled)

    for name in column_names:
        meanx = training_inputs[name].mean()
        #print name,meanx
        stdx                                = training_inputs[name].std()
        testing_inputs_scaled[name       ] -= meanx
        testing_inputs_scaled[name       ] /= stdx

    testing_inputs_scaled  = testing_inputs_scaled.replace(np.nan,0)
    testing_inputs_scaled.to_csv('scaled_inputs_new.csv')
    testing_inputs_scaled  = testing_inputs_scaled.as_matrix()

    json_file = open(path+str(i)+'_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+str(i)+"_model.h5")
    print("Loaded model from disk")
 
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    prediction = loaded_model.predict(testing_inputs_scaled)
    print prediction
    predx      = [round(x[1]) for x in prediction]
    #print predx
    max_pred = []
    
    for _ in prediction:
        max_pred.append(max(_))
    
    return (predx,max_pred)