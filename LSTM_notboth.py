import Data_load as dl
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from time import time

current_path = os.getcwd()

DataX = list()
DataY = list()

sensor = 3

total_epoch = 1000
#Xlength = 0

#Xlength = DataX.__len__() - Xlength
#print(Xlength)
'''
model = keras.Sequential()
model.add(keras.layers.LSTM(128, return_sequences=True, input_shape=(1,100)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.LSTM(128, return_sequences=True,))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.LSTM(128))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(5, activation='relu'))

model.summary()
'''
model = keras.Sequential()
model.add(keras.layers.LSTM(128, input_shape=(1,100)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(5, activation='relu'))
'''
model = keras.Sequential()
model.add(keras.layers.LSTM(128, activation='tanh',input_shape=(1,100), return_sequences=False))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(6))
model.add(keras.layers.LeakyReLU())
'''
'''
model = keras.Sequential()
model.add(keras.layers.LSTM(128, input_shape=(1,100)))
model.add(keras.layers.Dense(5, activation='relu'))
'''
'''
model = keras.Sequential()
model.add(keras.layers.LSTM(128, input_shape=(1,100)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(6, activation='relu'))
'''
'''
inputs = keras.Input(shape=(1,100))

x = keras.layers.LSTM(128, return_sequences=True)(inputs)
x = keras.layers.LSTM(128, return_sequences=True)(x)
x = keras.layers.LSTM(6)(x)

model = keras.models.Model(inputs, x)
'''

modelname = input("Input Model name :: ")

dl.Data_load_notboth("Sunny", "Animal", sensor, DataX, DataY)
dl.Data_load_notboth("Sunny", "Human", sensor, DataX, DataY)
dl.Data_load_notboth("Sunny", "Both", sensor, DataX, DataY)
dl.Data_load_notboth("Sunny", "None", sensor, DataX, DataY)
dl.Data_load_notboth("Rain", "Animal", sensor, DataX, DataY)
dl.Data_load_notboth("Rain", "Human", sensor, DataX, DataY)
dl.Data_load_notboth("Rain", "Both", sensor, DataX, DataY)
dl.Data_load_notboth("Rain", "None", sensor, DataX, DataY)

DataX = np.asarray(DataX)
DataY = np.asarray(DataY)

print(DataX.shape)
print(DataY.shape)

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
              loss='mean_squared_error',
              #loss='binary_crossentropy',
              metrics=['acc'])

X_train, X_test, Y_train, Y_test = train_test_split(DataX, DataY, test_size=0.3, random_state=0, shuffle=True)
#X_train, X_test, Y_train, Y_test = train_test_split(DataX, DataY, test_size=0.2, random_state=321, shuffle=True)
#X_train, X_test, Y_train, Y_test = train_test_split(DataX, DataY, test_size=0.2, random_state=777, shuffle=True)

k = 0
for i in range(0, DataY.__len__()):
    if DataY[i][2] == 1:
        k = k + 1

print("None 데이터셋 갯수 :: " + str(k))

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

print(X_train.shape)
print(Y_train.shape)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto')
model.fit(X_train, Y_train, epochs=total_epoch, batch_size=256, shuffle='True', validation_data=(X_test, Y_test), verbose=1, callbacks=[tensorboard,early_stopping])

x_validate = X_train[:10000]
y_validate = Y_train[:10000]
print(y_validate)

results = model.evaluate(x_validate, y_validate)

print("Validate data[loss, accuracy] :: ")
print(results)

model_json = model.to_json()

modelname_json = "Models/" + modelname + ".json"
modelname_weight = "Models/" + modelname + ".h5"

with open(modelname_json, "w") as json_file :
    json_file.write(model_json)

model.save_weights(modelname_weight)
print(modelname)
print("Model Saved...!")