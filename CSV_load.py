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
model.add(keras.layers.LSTM(32, input_shape=(sensor_num ,hz)))
model.add(keras.layers.Dense(2))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['acc'])

label = input("(1. Nobody // 2.Somebody ) : ")
date = input("Insert date : ")
label_tag = ''

if label == '1':
    csv_path = csv_nobody_path
    y_label = [0,1] # nobody
    label_tag = 'none'
elif label == '2':
    csv_path = csv_somebody_path
    y_label = [1,0] # somebody
    label_tag = 'human'


X = list()
Y = list()

def find_filenum(sensor_num):
    csv_path_sensor = csv_path + '/' + str(sensor_num) + '/'
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
            exist = True
        else:
            exist = False

    if(exist):
        for k in range(0, (len(data[0]))):
           for h in range(0, sensor_num):
                X.append(data[h][k])

    Y.append(y_label)

x_data = np.asarray(X)
y_data = np.asarray(Y)
data_num = (int)(x_data.size/3)
x_data = np.reshape(x_data, [data_num, 3])


'''
def find_filenum(sensor_num):
    csv_path_sensor = csv_path + '/' + str(sensor_num) + '/'
    return len(next(os.walk(csv_path_sensor))[2])


# 3 directory's file number is same.(if)
file_num = find_filenum(1)
file_count = 0

X = list()

for i in range(1, file_num):
    for j in range(1, sensor_num + 1):
        #print(" i is ..... ::: " + str(i))
        #print(" j is ..... ::: " + str(j))
        os.chdir(path_current)
        os.chdir(csv_path + '/' + str(j) + '/')
        file_name = 'labeled_data' + str(j) + '_' + date + '_' + label_tag + str(i) + '.csv'

        if(os.path.exists(file_name)):
            data = np.loadtxt(file_name, delimiter=",", dtype=np.float32)

        x_data = np.array(data.size/hz)

        if(os.path.exists(file_name)):
            x_data = np.array(data)
            for k in range(0, int(data.size/hz)):
                x_data = np.vstack((x_data,data[k]))
                file_count += 1

        X.append(x_data)

    #x_data = x_data.reshape(int(data.size/hz), sensor_num, hz)

print(X[0])



print(";;;;;;;;;;;;;;;;;;;;;;")



csv_path = 'PIR_data/new_csv/nobody/'

def csv_file_load(pir_number, date, label):
    return 'labeled_data' + str(pir_number) + '_' + date + '_' + label + '.csv'

filename = csv_file_load(1,'20180814_102745','none1')
filename2 = csv_file_load(2,'20180814_102745','none1')
filename3 = csv_file_load(3,'20180814_102745','none1')

print(filename)
print(filename2)
print(filename3)

os.chdir(path_current)
os.chdir(csv_path +'1/')
data1 = np.loadtxt(filename, delimiter=",", dtype=np.float32)
os.chdir(path_current)
os.chdir(csv_path +'2/')
data2 = np.loadtxt(filename2, delimiter=",", dtype=np.float32)
os.chdir(path_current)
os.chdir(csv_path +'3/')
data3 = np.loadtxt(filename3, delimiter=",", dtype=np.float32)

data_size = int(data1.size / 100)
x_data = np.empty(((data_size), sensor_num , hz), dtype=np.float32)
y_data = np.zeros(((data_size), 2), dtype=np.float32)

for i in range(0,data_size):
    x_data[i] = np.vstack((data1[i],data2[i],data3[i]))
    y_data[i] = y_label


print(x_data.shape)

X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=321)

history = model.fit(X_train,
                    Y_train,
                    epochs=5,
                    batch_size=512,
                    validation_data=(X_test, Y_test),
                    verbose=1)

results = model.evaluate(X_test, Y_test)

print(results)

print(model.predict_classes(X_test[:1, :], verbose=0))
print('----------------------------------------------')
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)



'''
