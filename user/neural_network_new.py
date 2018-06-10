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
from keras.models import load_model
path = "./neural_network/"
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

    #json_file = open(path+str(i)+'_model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    ## load weights into new model
    #loaded_model.load_weights(path+str(i)+"_model.h5")
    #print("Loaded model from disk")
    model = load_model(path+str(i)+"_my_model.h5")
    # evaluate loaded model on test data
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy', matthews_correlation])
    prediction = model.predict(testing_inputs_scaled)
    print prediction
    predx      = [round(x[1]) for x in prediction]
    #print predx
    max_pred = []
    
    for _ in prediction:
        max_pred.append(max(_))
    
    return (predx,max_pred)