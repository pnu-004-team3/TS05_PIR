import Data_load_4 as dl
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

#Xlength = DataX.__len__() - Xlength
#print(Xlength)
'''
model = keras.Sequential()
model.add(keras.layers.LSTM(128, input_shape=(4,100)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(3, activation='relu'))

model.summary()
'''
'''
model = keras.Sequential()
model.add(keras.layers.LSTM(128, return_sequences=True, input_shape=(4,100), activation='relu', recurrent_activation="sigmoid"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.LSTM(128, return_sequences=True, activation='relu', recurrent_activation="sigmoid"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.LSTM(128, activation='relu', recurrent_activation="sigmoid"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(3, activation='relu'))
'''
modelname = input("Input model name to save :: ")

totalepoch = 2000

model = keras.Sequential()
model.add(keras.layers.LSTM(128, input_shape=(4,100)))
model.add(keras.layers.Dense(2, activation='relu'))

## Data 순서 :: pir , light, temp , humid
dl.Data_load("None", DataX, DataY)

#dl.Data_load("Animal", DataX, DataY)

dl.Data_load("Human", DataX, DataY)

DataX = np.asarray(DataX)
DataY = np.asarray(DataY)

print(DataX.shape)
print(DataY.shape)

DataX = np.reshape(DataX, (int(DataX.__len__()/4), 4, 100))
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['acc'])

X_train, X_test, Y_train, Y_test = train_test_split(DataX, DataY, test_size=0.3, random_state=777, shuffle=True)
#X_train, X_test, Y_train, Y_test = train_test_split(DataX, DataY, test_size=0.2, random_state=321, shuffle=True)
#X_train, X_test, Y_train, Y_test = train_test_split(DataX, DataY, test_size=0.2, random_state=0, shuffle=True)

k = 0
for i in range(0, DataY.__len__()):
    if DataY[i][0] == 1:
        k = k + 1

print("None 데이터셋 갯수 :: " + str(k))

print(X_train.shape)
print(Y_train.shape)

#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=0, mode='auto')
model.fit(X_train, Y_train, epochs=totalepoch, batch_size=512, shuffle='True', validation_data=(X_test, Y_test), verbose=1, callbacks=[tensorboard])

x_validate = X_train[:3000]
y_validate = Y_train[:3000]
print(y_validate)

results = model.evaluate(x_validate, y_validate)

print("Validate data[loss, accuracy] :: ")
print(results)

print(model.predict_classes(X_test[:1, :], verbose=0))
print('----------------------------------------------')

model_json = model.to_json()

with open("Models/" + modelname + ".json", "w") as json_file :
    json_file.write(model_json)

model.save_weights("Models/" + modelname + ".h5")



print(modelname)
print("Model Saved...!")

print("Train Dataset .. :: ")
print(X_train.shape)
print("Validate Dataset .. :: ")
print(X_test.shape)
