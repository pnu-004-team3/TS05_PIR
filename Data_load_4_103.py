import numpy as np
import os

# 데이터 순서 --> pir , light , temp , humid

def Data_load(exist, X, Y):

    current_path = os.getcwd()

    classfication = Labeling_tag(exist)
    print(classfication)
    os.chdir("Labeled_data_4/" + exist + "/")

    typepath = os.getcwd()
    os.chdir("PIR/")

    for filename in os.listdir("."):
        print("filename ::: " + filename)
        fdate = filename.split("_")

        fpirn = filename
        ftempn = ""
        flightn = ""
        fhumidn = ""

        os.chdir(typepath)
        os.chdir("TEMP/")
        for typelist in os.listdir("."):
            filetyname = typelist.split("_")
            if filetyname[2:5] == fdate[2:5]:
                ftempn = typelist
                print(ftempn)
                break

        os.chdir(typepath)
        os.chdir("LIGHT/")
        for typelist in os.listdir("."):
            filetyname = typelist.split("_")
            if filetyname[2:5] == fdate[2:5]:
                flightn = typelist
                print(flightn)
                break

        os.chdir(typepath)
        os.chdir("HUMID/")
        for typelist in os.listdir("."):
            filetyname = typelist.split("_")
            if filetyname[2:5] == fdate[2:5]:
                fhumidn = typelist
                print(fhumidn)
                break

        os.chdir(typepath)

        data_pir = np.loadtxt("PIR/" + fpirn, delimiter=",", dtype=np.float32)
        data_temp = np.loadtxt("TEMP/" + ftempn, delimiter=",", dtype=np.float32)
        data_light = np.loadtxt("LIGHT/" + flightn, delimiter=",", dtype=np.float32)
        data_humid = np.loadtxt("HUMID/" + fhumidn, delimiter=",", dtype=np.float32)
        k = 0
        for i in range(0, data_pir.__len__()):
            returnX = list()
            for j in range(0, 100):
                returnX.append(data_pir[i][j])
            returnX.append(data_light[i][0])
            returnX.append(data_temp[i][0])
            returnX.append(data_humid[i][0])

            X.append(returnX)
            Y.append(classfication)

        '''
        fsize = 0
        for typelist in os.listdir("."):
            filetyname = typelist.split("_")
            if filetyname[2] == fdate[2] and filetyname[3] == fdate[3]:
                print(filetyname)
                data = np.loadtxt(filename, delimiter=",", dtype=np.float32)
                fsize = data.__len__()

            data.reshape(fsize, 100)
            for i in range(0, data.__len__()):
                X.append(data[i])
                Y.append(classfication)
        '''
    os.chdir(current_path)

def Labeling_tag(exist):
    if exist == "None":
        return [1, 0]
    #elif exist == "Animal":
    #    return [0, 1, 0]
    elif exist == "Human":
        return [0, 1]