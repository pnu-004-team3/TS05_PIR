import numpy as np
import os

def Data_load(weather, exist, sensor_use, X, Y):

    os.chdir(current_path)
    classfication = [weather,exist]

    os.chdir("labeled_data/" + weather + "/" + exist + "/" + str(sensor_use) + "/")

    print(os.getcwd())

    for filename in os.listdir("."):
        #print(filename)
        data = np.loadtxt(filename, delimiter=",", dtype=np.float32)

        for i in range(0, data.__len__()):
            X.append(data[i])
            Y.append(classfication)


def Data_setY(length, classfication,Y):

    for i in range(0, length):
        Y.append(classfication)


current_path = os.getcwd()

DataX = list()
DataY = list()

Data_load("Sunny", "Animal", 2, DataX, DataY)
print(DataX.__len__())
print(DataY.__len__())

Data_load("Sunny", "Human", 2, DataX, DataY)
print(DataX.__len__())
print(DataY.__len__())

Data_load("Sunny", "Both", 2, DataX, DataY)
print(DataX.__len__())
print(DataY.__len__())

Data_load("Sunny", "None", 2, DataX, DataY)
print(DataX.__len__())
print(DataY.__len__())

DataX = np.asarray(DataX)
DataY = np.asarray(DataY)

print(DataX.shape)
print(DataY.shape)