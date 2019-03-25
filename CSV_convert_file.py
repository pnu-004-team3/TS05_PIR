import os

path_nobody = 'PIR_data/nobody/'
path_somebody = 'PIR_data/somebody/'
path_change_nobody = 'PIR_data/new_csv/nobody'
path_change_somebody = 'PIR_data/new_csv/somebody'

path_current = os.getcwd()

sensor_count = 3

for root, dirs, files in os.walk(path_nobody + str(sensor_count) + '/'):

    for fname in files:
        os.chdir(path_current)
        os.chdir(path_nobody + str(sensor_count) + '/')

        with open(fname, 'r') as f:
            data = f.read()
            data = data.replace(' ', ',')

        os.chdir(path_current)
        os.chdir(path_change_nobody + '/' + str(sensor_count) + '/')

        with open(fname, 'w') as f:
            f.write(data)

    print('Sensor ' + str(sensor_count) + ' data is converted .. !')


for root, dirs, files in os.walk(path_somebody + str(sensor_count) + '/'):

    for fname in files:
        os.chdir(path_current)
        os.chdir(path_somebody + str(sensor_count) + '/')

        with open(fname, 'r') as f:
            data = f.read()
            data = data.replace(' ', ',')

        os.chdir(path_current)
        os.chdir(path_change_somebody + '/' + str(sensor_count) + '/')

        with open(fname, 'w') as f:
            f.write(data)

    print('Sensor ' + str(sensor_count) + ' data is converted .. !')