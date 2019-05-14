import os
import numpy as np
import pandas as pd

def file_exist(sensor_num, file_name):

    for filename in os.listdir("."):
        if filename.startswith("pir" + str(sensor_num) + file_name):
            print(filename)
            return True

    return False


while (True):
    date = input("Write the date (Month-day) (if insert 0, exit) :: ")
    if (date == "0"):
        break;
    time = input("Write the time (HH-MM-SS) :: ")
    pre_file_name = "_data_" + date + "_" + time

    current_path = os.getcwd()
    sr = 15

    while (True):
        month = date[:2]
        sensor_num = 3

        if (month == "01"):
            month = "Jan"
        elif (month == "02"):
            month = "Feb"
        elif (month == "03"):
            month = "March"
        elif (month == "04"):
            month = "April"

        os.chdir(current_path)
        os.chdir(month + "/")

        file_name = pre_file_name.split('-')
        file_name = '-'.join(file_name[:2])
        print("pir1" + file_name)

        pir_exist = file_exist(1,file_name) and file_exist(2,file_name) and file_exist(3,file_name)

        if (pir_exist == True):

            label_exist = input("1. Exist  /   2. None  :: ")
            if (label_exist == "2"):
                tag_exist = "None"
            elif (label_exist == "1"):
                label_who = input("1. Human  /   2. Animal  /  3. Both :: ")
                if (label_who == "1"):
                    tag_exist = "Human"
                elif (label_who == "2"):
                    tag_exist = "Animal"
                elif (label_who == "3"):
                    tag_exist = "Both"
                elif (label_who == "0"):
                    break
            elif (label_exist == "0"):
                break

            label_weather = input("1. Sunny  /  2. Rain  :: ")
            if (label_weather == "1"):
                tag_weather = "Sunny"
            elif (label_weather == "2"):
                tag_weather = "Rain"
            # elif (label_weather == "3"):
            #    tag_weather = "Cloud"
            elif (label_weather == "0"):
                break

            while (True):
                sn_start = input("Insert start sequential number ::  ")
                sn_end = input("Insert end sequential number :: ")

                sn_start = int(sn_start)
                sn_end = int(sn_end)

                if (sn_start >= sn_end):
                    print("sn_start :: " + str(sn_start))
                    print("sn_end :: " + str(sn_end))
                    print("Wrong Sequential Number .. !")
                else:
                    break

            for i in range(1, sensor_num + 1):
                os.chdir(current_path)
                os.chdir(month + "/")
                for filename in os.listdir("."):
                    if filename.startswith("pir" + str(i) + file_name):
                        if i == 1:
                            main_fname = filename.split(".")[0]
                            main_fname = main_fname.split("_")[1:]
                            main_fname = "_".join(main_fname)
                            main_fname = "pir" + str(i) + "_" + main_fname
                        else:
                            main_fname = "pir" + str(i) + main_fname[4:]

                        ofname = filename
                        break

                fname = "labeled_" + main_fname + "_" + tag_exist

                data = np.loadtxt(ofname)

                os.chdir(current_path)
                os.chdir("labeled_data/" + tag_weather + "/" + tag_exist + "/")
                sensor_path = os.getcwd()

                os.chdir(str(i) + "/")
                k = 0
                while (True):
                    if (os.path.exists(fname + str(k) + ".csv")):
                        k = k + 1
                    else:
                        fname = fname + str(k) + ".csv"
                        break

                X = list()
                for j in range(sn_start, sn_end, sr):
                    if (j >= sn_end):
                        break

                    slice_data = data[j - 1: j - 1 + 100]
                    np.asarray(slice_data)
                    X.append(slice_data)

                dataframe = pd.DataFrame(X)
                dataframe.to_csv(fname, header=False, index=False)

                print(fname)
                os.chdir(sensor_path)
        else:
            print("File is not Exist..!")
            os.chdir(current_path)
            break

    os.chdir(current_path)


