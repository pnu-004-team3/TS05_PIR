import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import sys

sys.getdefaultencoding()

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


csv_nobody_path = 'PIR_data/new_csv/nobody/'
csv_somebody_path = 'PIR_data/new_csv/somebody/'
path_current = os.getcwd()
csv_path = ''

sensor_num = 3
hz = 100

model = keras.Sequential()
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(sensor_num ,hz)))
#model.add(keras.layers.LSTM(64, input_shape=(sensor_num ,hz)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.LSTM(32, return_sequences=True))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.LSTM(16))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mean_squared_error',
              #loss='binary_crossentropy',
              metrics=['acc'])

X = list()
Y = list()

while(True):
    menu_label = input("(1. Nobody // 2.Somebody // 3. Learning) : ")

    label_tag = ''
    if menu_label == '1':
        csv_path = csv_nobody_path
        y_label = [0, 1]  # nobody
        label_tag = 'none'
    elif menu_label == '2':
        csv_path = csv_somebody_path
        y_label = [1, 0]  # somebody
        label_tag = 'human'
    elif menu_label == '3':
        break;

    print(y_label)
    print(label_tag)
    date = input("Insert date : ")

    def find_filenum(sensor_num):
        os.chdir(path_current)
        csv_path_sensor = csv_path + str(sensor_num) + '/'
        return len(next(os.walk(csv_path_sensor))[2])

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
                file_count += 1
                exist = True
            else:
                exist = False

        if (exist):
            for k in range(0, (len(data[0]))):
                for h in range(0, sensor_num):
                    X.append(data[h][k])

                Y.append(y_label)

    print(len(X))

    print((str)(file_count) + " files opened")
    print((str)(len(X)/sensor_num) + " dataes inserted")

os.chdir(path_current)

x_data = np.asarray(X)
y_data = np.asarray(Y)

print(x_data.shape)
print(y_data.shape)

x_data = np.reshape(x_data, [(int)(len(x_data) / sensor_num), sensor_num, hz])

x_validate = x_data[:100]
y_validate = y_data[:100]


X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=321)

history_tensorboard = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(X_train, Y_train, epochs=1000, batch_size=128, shuffle='True', validation_data=(X_test, Y_test), verbose=1, callbacks=[history_tensorboard])

results = model.evaluate(x_validate, y_validate)

print("Validate data[loss, accuracy] :: ")
print(results)

print(model.predict_classes(X_test[:1, :], verbose=0))
print('----------------------------------------------')

'''
history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.plot(epochs, loss, 'ro', label="Training loss")
plt.plot(epochs, val_loss, 'r', label="Validation loss")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
'''

#model.save("Mymodel")

'''
print("Layer[0] ::")
print(len(model.layers[0].get_weights()))
print("Layer[1] ::")
print(model.layers[1].get_weights())
print("Layer[2] ::")
print(model.layers[2].get_weights())



#print(X_train)
#print(Y_train)
#print(X_test)
#print(Y_test)
'''
