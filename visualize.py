from keras.utils import *
import tensorflow as tf
import numpy as np
import keras
import random
from keras.models import model_from_json
import Data_load_testdata as dl
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import seaborn as sns
import Data_load_testdata as dlt
'''
modelname = "Model_cnn_onlypir_ep1000"
modelname_json = "Models/" + modelname + ".json"
modelname_weight = "Models/" + modelname + ".h5"
json_file = open(modelname_json,"r")

loaded_model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights(modelname_weight)
print("Model Loaded...! :: " + modelname)
'''


model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), input_shape=(10, 10, 1), data_format='channels_last', activation='relu'))
model.add(keras.layers.Conv2D(1, kernel_size=(3,3),activation='relu'))

model2 = keras.Sequential()

model2.add(keras.layers.Conv2D(32, kernel_size=(3,3), input_shape=(10, 10, 1), data_format='channels_last', activation='relu'))
model2.add(keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'))
model2.add(keras.layers.Conv2D(1, kernel_size=(3,3),activation='relu'))

exist = list()
existy = list()
none = list()
noney = list()

dlt.Data_load("Human", exist, existy)
dlt.Data_load("None", none, noney)

exist = np.asarray(exist)
none = np.asarray(none)

exist = np.reshape(exist, (int(len(exist)/4), 4, 100))
none = np.reshape(none, (int(len(none)/4), 4, 100))

data = list()
datan = list()

for i in range(len(exist)):
    data.append(exist[i][0])

for j in range(len(none)):
    datan.append(none[i][0])

data = np.reshape(data, (-1, 1, 10, 10, 1))
datan = np.reshape(datan, (-1, 1,  10, 10, 1))

conv_cat = model.predict(data[300])
not_cat = model.predict(datan[300])

def nice_cat_printer(model, cat):

    conv_cat2 = model.predict(cat)

    conv_cat2 = np.squeeze(conv_cat2, axis=0)
    print(conv_cat2.shape)
    conv_cat2 = conv_cat2.reshape((conv_cat2.shape[:2]))
    print(conv_cat2.shape)
    plt.imshow(conv_cat2)
    plt.show()

def visualize_cat(cat_batch):
    cat = np.squeeze(cat_batch, axis=0)
    print(cat.shape)
    plt.imshow((cat* 255).astype(np.uint8))
    plt.show()

nice_cat_printer(model,data[20])
#nice_cat_printer(model2,data[11])
nice_cat_printer(model,datan[20])
#nice_cat_printer(model2,datan[11])
#visualize_cat(conv_cat)