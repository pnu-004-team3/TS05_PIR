from keras.utils import *
import tensorflow as tf
import numpy as np
import random
from keras.models import model_from_json
import Data_load_testdata as dlt
import Data_load_4 as dl
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import seaborn as sns
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
'''
#modelname = input("Type Model's name :: ")
modelname = "Model_onlypir_lstm_ep2000"
modelname_json = "Models/" + modelname + ".json"
modelname_weight = "Models/" + modelname + ".h5"
json_file = open(modelname_json,"r")

loaded_model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights(modelname_weight)
print("Model Loaded...! :: " + modelname)

DataX = list()
DataY = list()

dl.Data_load("None", DataX, DataY)

dl.Data_load("Human", DataX, DataY)

dlt.Data_load("None", DataX, DataY)

dlt.Data_load("Human", DataX, DataY)

exist_none = list()
exist_exist = list()
none_none = list()
none_exist = list()

DataX = np.asarray(DataX)
DataY = np.asarray(DataY)

DataX = np.reshape(DataX, (int(DataX.__len__()/4), 4, 100))

x_data =list()

def decision(predict):
    if predict[0][0] > predict[0][1]:
        return "None"
    else:
        return "Exist"

def predict_result(x_data, result, y_data,):
    global exist_exist
    global exist_none
    global none_exist
    global none_none

    #say exist , but none
    if result == "Exist" and y_data[0] == 1:
        exist_none.append(x_data[0])
    # say exist , and exist
    elif result == "Exist" and y_data[0] == 0:
        exist_exist.append(x_data[0])
    # say none , and none
    elif result == "None" and y_data[0] == 1:
        none_none.append(x_data[0])
    # say none, but exist
    elif result == "None" and y_data[0] == 0:
        none_exist.append(x_data[0])

for i in range(len(DataX)):
    x_data.append(DataX[i][0])

x_data = np.asarray(x_data)
x_data = np.reshape(x_data, (-1, 1, 100))

X_train, X_test, Y_train, Y_test = train_test_split(x_data, DataY, test_size=0.3, random_state=777, shuffle=True)

for i in range(len(X_test)):
    testdata = X_test[i]
    testdata = np.reshape(testdata, (1, 1, 100))
    result = decision(model.predict(testdata))
    predict_result(testdata, result, Y_test[i])

    if i % 1000 == 0:
        print(i)

ee = list()
en = list()
ne = list()
nn = list()

for i in range(0, 1000):
    if i < len(exist_exist):
        ee.append(exist_exist[i][0])
    if i < len(exist_none):
        en.append(exist_none[i][0])
    if i < len(none_exist):
        ne.append(none_exist[i][0])
    if i < len(none_none):
        nn.append(none_none[i][0])

print("Exist _ Exist :: " + str(len(exist_exist)))
print("Exist _ None :: " + str(len(exist_none)))
print("None _ Exist :: " + str(len(none_exist)))
print("None _ None :: " + str(len(none_none)))

#plt.title("")
for i in range(20):
    print(ne[i])


for j in range(20):
    print(nn[j])
sns.tsplot(data=ee, color="blue")
sns.tsplot(data=en, color="red")
#sns.tsplot(data=ne, color="black")
#sns.tsplot(data=nn, color="yellow")

plt.show()