import numpy as np
import os
import tensorflow as tf
import sys

path_sub = 'labeled_data_sub/'
path_origin = 'labeled_data/'

path_current = os.getcwd()

sensor_num = 3

def convert_data(path, sensor_num):

    os.chdir(path_origin + path + str(sensor_num) + "/")

    for filename in os.listdir("."):

        os.chdir(path_current)
        os.chdir(path_origin + path + str(sensor_num) + "/")

        print(filename)
        data = np.loadtxt(filename, delimiter=",", dtype=np.float32)

        os.chdir(path_current)
        os.chdir(path_sub + path + str(sensor_num) + "/")

        with open(filename, 'a') as f:
            for i in range(0, data.__len__()):

                for j in range(0, 99):
                    if j == 98:
                        f.write(str(np.abs(data[i][j] -data[i][j+1])) + "\n")
                    else:
                        f.write(str(np.abs(data[i][j] -data[i][j+1])) + ",")


    os.chdir(path_current)


#convert_data('rain/animal/', 3)
#convert_data('rain/both/', 3)
#convert_data('rain/human/', 3)
#convert_data('rain/none/', 3)

convert_data('sunny/animal/', 3)
convert_data('sunny/both/', 3)
convert_data('sunny/human/', 3)
convert_data('sunny/none/', 3)