import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

csv_nobody_path = 'PIR_data/new_csv/nobody/'
csv_somebody_path = 'PIR_data/new_csv/somebody/'
path_current = os.getcwd()
csv_path = ''

sensor_num = 3
hz = 100

model = keras.Sequential()
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(sensor_num ,hz)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['acc'])


menu_label = input("(1. Nobody // 2.Somebody // 3. Learning) : ")
date = input("Insert date : ")
label_tag = ''

if menu_label == '1':
    csv_path = csv_nobody_path
    y_label = [0, 1]  # nobody
    label_tag = 'none'
elif menu_label == '2':
    csv_path = csv_somebody_path
    y_label = [1, 0]  # somebody
    label_tag = 'human'


def find_filenum(sensor_num):
    csv_path_sensor = csv_path + '/' + str(sensor_num) + '/'
    return len(next(os.walk(csv_path_sensor))[2])

X = list()
Y = list()

file_num = find_filenum(1)
file_count = 0

for i in range(1, file_num):
    data = list()
    for j in range(1, sensor_num + 1):
        # print(" i is ..... ::: " + str(i))
        # print(" j is ..... ::: " + str(j))
        os.chdir(path_current)
        os.chdir(csv_path + '/' + str(j) + '/')
        file_name = 'labeled_data' + str(j) + '_' + date + '_' + label_tag + str(i) + '.csv'

        if (os.path.exists(file_name)):
            data.append(np.loadtxt(file_name, delimiter=",", dtype=np.float32))
            exist = True
        else:
            exist = False

    if (exist):
        for k in range(0, (len(data[0]))):

            for h in range(0, sensor_num):
                X.append(data[h][k])

            Y.append(y_label)

x_data = np.asarray(X)
y_data = np.asarray(Y)

x_data = np.reshape(x_data, [(int)(len(x_data) / sensor_num), sensor_num, hz])

x_validate = x_data[:100]
y_validate = y_data[:100]

print(x_data.shape)
print(y_data.shape)

X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=321)

history = model.fit(X_train,
                    Y_train,
                    epochs=10,
                    batch_size=100,
                    shuffle='True',
                    validation_data=(X_test, Y_test),
                    verbose=1)

results = model.evaluate(x_validate, y_validate)

print(results)

print(model.predict_classes(X_test[:1, :], verbose=0))
print('----------------------------------------------')
#print(X_train)
#print(Y_train)
#print(X_test)
#print(Y_test)