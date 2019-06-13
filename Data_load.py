import numpy as np
import os

def Data_load(weather, exist, sensor_use, X, Y):

    current_path = os.getcwd()

    classfication = Labeling_tag(weather, exist)
    print(classfication)
    os.chdir("labeled_data/" + weather + "/" + exist + "/" + str(sensor_use) + "/")

    print(os.getcwd())

    for filename in os.listdir("."):
        print(filename)
        data = np.loadtxt(filename, delimiter=",", dtype=np.float32)

        for i in range(0, data.__len__()):
            X.append(data[i])
            Y.append(classfication)
            if classfication[2] == 1 and i > 80 :
                print("Cut.. ! (>80)")
                break

    os.chdir(current_path)

def Data_load_notweather(exist, sensor_use, X, Y):
    current_path = os.getcwd()

    classfication = Labeling_tag_notweather(exist)
    print(classfication)
    os.chdir("labeled_data_notweather/" + exist + "/" + str(sensor_use) + "/")

    print(os.getcwd())

    for filename in os.listdir("."):
        print(filename)
        data = np.loadtxt(filename, delimiter=",", dtype=np.float32)

        for i in range(0, data.__len__()):
            X.append(data[i])
            Y.append(classfication)
            if classfication[2] == 1 and i > 80:
                print("Cut.. ! (>80)")
                break

    os.chdir(current_path)


def Data_load_notboth(weather,exist, sensor_use, X, Y):

    current_path = os.getcwd()

    classfication = Labeling_tag_notboth(weather, exist)
    print(classfication)
    os.chdir("labeled_data/" + weather + "/" + exist + "/" + str(sensor_use) + "/")

    print(os.getcwd())

    for filename in os.listdir("."):
        print(filename)
        data = np.loadtxt(filename, delimiter=",", dtype=np.float32)

        for i in range(0, data.__len__()):
            X.append(data[i])
            Y.append(classfication)
            if classfication[2] == 1 and i > 80:
                print("Cut.. ! (>80)")
                break

    os.chdir(current_path)


def Labeling_tag_notweather(exist):
    if exist == "None":
        return [1, 0, 0, 0]
    elif exist == "Animal":
        return [0, 1, 0, 0]
    elif exist == "Both":
        return [0, 0, 1, 0]
    elif exist == "Human":
        return [0, 0, 0, 1]

def Labeling_tag_notboth(weather, exist):
    if weather == "Sunny":
        if exist == "Animal":
            return [1, 0, 0, 1, 0]
        elif exist == "Both" or exist =="Human":
            return [1, 0, 0, 0, 1]
        elif exist == "None":
            return [1, 0, 1, 0, 0]

    elif weather == "Rain":
        if exist == "Animal":
            return [0, 1, 0, 1, 0]
        elif exist == "Both" or exist == "Human":
            return [0, 1, 0, 0, 1]
        elif exist == "None":
            return [0, 1, 1, 0, 0]

def Labeling_tag(weather, exist):
    if weather == "Sunny":
        if exist == "Animal":
            return [1, 0, 0, 1, 0, 0]
        elif exist == "Both":
            return [1, 0, 0, 0, 1, 0]
        elif exist == "Human":
            return [1, 0, 0, 0, 0, 1]
        elif exist == "None":
            return [1, 0, 1, 0, 0, 0]

    elif weather == "Rain":
        if exist == "Animal":
            return [0, 1, 0, 1, 0, 0]
        elif exist == "Both":
            return [0, 1, 0, 0, 1, 0]
        elif exist == "Human":
            return [0, 1, 0, 0, 0, 1]
        elif exist == "None":
            return [0, 1, 1, 0, 0, 0]

def Label_tag(weather, exist):
    if weather == "Sunny":
        if exist == "Animal":
            return [0, 1, 0, 0, 0, 0, 0, 0]
        elif exist == "Both":
            return [0, 0, 1, 0, 0, 0, 0, 0]
        elif exist == "Human":
            return [0, 0, 0, 1, 0, 0, 0, 0]
        elif exist == "None":
            return [1, 0, 0, 0, 0, 0, 0, 0]

    elif weather == "Rain":
        if exist == "Animal":
            return [0, 0, 0, 0, 0, 1, 0, 0]
        elif exist == "Both":
            return [0, 0, 0, 0, 0, 0, 1, 0]
        elif exist == "Human":
            return [0, 0, 0, 0, 0, 0, 0, 1]
        elif exist == "None":
            return [0, 0, 0, 0, 1, 0, 0, 0]