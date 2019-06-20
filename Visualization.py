from keras.utils import *
import tensorflow as tf
import numpy as np
import random
from keras.models import model_from_json
import Data_load_testdata as dl
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import seaborn as sns

modelname = "Model_cnn_merge_predict"
modelname_json = "Models/" + modelname + ".json"
modelname_weight = "Models/" + modelname + ".h5"
json_file = open(modelname_json,"r")

loaded_model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights(modelname_weight)
print("Model Loaded...! :: " + modelname)

data = np.array([[
2476.0,2408.0,2298.0,2146.0,2072.0,2132.0,2134.0,2182.0,2088.0,2125.0,2102.0,2024.0,1912.0,1760.0,1692.0,1760.0,1806.0,1872.0,1828.0,1868.0,1840.0,1782.0,1680.0,1544.0,1472.0,1529.0,1598.0,1576.0,1521.0,1724.0,1868.0,1787.0,1809.0,1736.0,1641.0,1688.0,1696.0,1712.0,1620.0,1614.0,1550.0,1465.0,1337.0,1192.0,1113.0,1144.0,1214.0,1307.0,1264.0,1312.0,1296.0,1232.0,1129.0,1000.0,937.0,1034.0,1092.0,1184.0,1129.0,1194.0,1195.0,1161.0,1080.0,972.0,934.0,1044.0,1124.0,1197.0,1200.0,1256.0,1245.0,1202.0,1112.0,1006.0,961.0,1076.0,1164.0,1253.0,1256.0,1241.0,1216.0,1298.0,1379.0,1392.0,1317.0,1539.0,1746.0,1704.0,1528.0,1433.0,1276.0,1101.0,896.0,740.0,650.0,707.0,728.0,761.0,717.0,791.0
]])
data_other = np.array([[3980,25,31]])

from tensorflow.python.keras import models
layer_outputs = [layer.output for layer in model.layers[0:]]

data = np.reshape(data, (1, 10, 10, 1))
activations = model.predict([data,data_other])
first_layer_activation = activations[0]
print(first_layer_activation)
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0:], cmap='viridis')
'''
layer_names = []

for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 10

for layer_name, layer_activation in zip(layer_names, activations):
'''