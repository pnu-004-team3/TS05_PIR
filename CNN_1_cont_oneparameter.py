import Data_load_4 as dl
import Data_load_testdata as dlt
import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Concatenate, Conv2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from time import time

current_path = os.getcwd()

DataX = list()
DataY = list()

#Xlength = DataX.__len__() - Xlength
#print(Xlength)

modelname = input("Input model name to save :: ")

totalepoch = 1000

inputPIR = Input(shape=(1, 10, 10))
inputOther = Input(shape=(1,))

model_x1 = Conv2D(32, kernel_size=(3,3), input_shape=(1,10,10),  data_format='channels_first', activation='relu')(inputPIR)
model_x1 = keras.layers.BatchNormalization()(model_x1)
model_x1 = keras.layers.MaxPooling2D(pool_size=(2,2))(model_x1)
model_x1 = Conv2D(64, kernel_size=(3,3), activation='relu')(model_x1)
model_x1 = keras.layers.BatchNormalization()(model_x1)
model_x1 = keras.layers.MaxPooling2D(pool_size=(2,2))(model_x1)
model_x1 = keras.layers.Flatten()(model_x1)

model_x2 = Dense(1, input_shape=(1,))(inputOther)

merged = Concatenate(axis=1)([model_x1, model_x2])

model_x3 = Dense(2, activation='softmax')(merged)

model = Model(inputs=[inputPIR, inputOther], outputs=model_x3)

model.summary()

## Data 순서 :: pir , light, temp , humid
#dl.Data_load("None", DataX, DataY)

#dl.Data_load("Human", DataX, DataY)

dlt.Data_load("None", DataX, DataY)

dlt.Data_load("Human", DataX, DataY)
DataX = np.asarray(DataX)
DataY = np.asarray(DataY)

print(DataX.shape)
print(DataY.shape)

DataX = np.reshape(DataX, (int(DataX.__len__()/4), 4, 100))

x_data = list()
y_data = list()
x_data_test = list()
y_data_test = list()
x_data_other = list()
x_data_other_test = list()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['acc'])

X_train, X_test, Y_train, Y_test = train_test_split(DataX, DataY, test_size=0.3, random_state=777, shuffle=True)

for i in range(len(X_train)):
    x_data.append(X_train[i][0])
    #Light
    x_data_other.append(X_train[i][1][0])
    #Temp
    x_data_other.append(X_train[i][2][0])
    #Humid
    x_data_other.append(X_train[i][3][0])

for i in range(len(X_test)):
    x_data_test.append(X_test[i][0])
    #Light
    #x_data_other_test.append(X_test[i][1][0])
    #Temp
    #x_data_other_test.append(X_test[i][2][0])
    #Humid
    x_data_other_test.append(X_test[i][3][0])

x_data = np.asarray(x_data)
x_data_other = np.asarray(x_data_other)

x_data = np.reshape(x_data, (-1, 1, 10, 10))
x_data_other = np.reshape(x_data_other, (-1, 1))
x_data_test = np.reshape(x_data_test, (-1, 1, 10, 10))
x_data_other_test = np.reshape(x_data_other_test, (-1,1))

k = 0
for i in range(0, DataY.__len__()):
    if DataY[i][0] == 1:
        k = k + 1

print("None 데이터셋 갯수 :: " + str(k))

print(X_train.shape)
print(Y_train.shape)

print(x_data.shape)
print(x_data_other.shape)
print(x_data_test.shape)

#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=0, mode='auto')
model.fit([x_data,x_data_other], Y_train, epochs=totalepoch, batch_size=256, shuffle='True', validation_data=([x_data_test,x_data_other_test], Y_test), verbose=1, callbacks=[tensorboard])

x_validate = x_data[:3000]
x_validate_other = x_data_other[:3000]
y_validate = Y_train[:3000]
print(y_validate)

results = model.evaluate([x_validate,x_validate_other], y_validate)

print("Validate data[loss, accuracy] :: ")
print(results)

#print(model.predict_classes(X_test[:1, :], verbose=0))
#print('----------------------------------------------')

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
