import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
from keras.callbacks import EarlyStopping

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
model.add(keras.layers.LSTM(128, return_sequences=True, input_shape=(sensor_num ,hz), recurrent_activation="sigmoid"))
#model.add(keras.layers.LSTM(64, input_shape=(sensor_num ,hz)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.LSTM(128, return_sequences=True, recurrent_activation="sigmoid"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.LSTM(128))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(2, activation='relu'))

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

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

model.fit(X_train, Y_train, epochs=1000, batch_size=128, shuffle='True', validation_data=(X_test, Y_test), verbose=1, callbacks=[early_stopping])

results = model.evaluate(x_validate, y_validate)

print("Validate data[loss, accuracy] :: ")
print(results)

print(model.predict_classes(X_test[:1, :], verbose=0))
print('----------------------------------------------')

#1 human 2 rain none
test_data = np.array([[
1831.0,2329.0,3008.0,2998.0,2340.0,1795.0,1496.0,1400.0,1567.0,1560.0,1828.0,2351.0,3004.0,2997.0,2251.0,1640.0,1292.0,1204.0,1384.0,1439.0,1756.0,2320.0,3011.0,2938.0,2384.0,1882.0,1579.0,1453.0,1599.0,1555.0,1784.0,2304.0,3000.0,3016.0,2723.0,2061.0,1596.0,1356.0,1420.0,1426.0,1666.0,2205.0,3000.0,3000.0,2366.0,1828.0,1538.0,1424.0,1582.0,1599.0,1828.0,2348.0,3012.0,3008.0,2360.0,1809.0,1508.0,1421.0,1578.0,1618.0,1816.0,2328.0,3013.0,3001.0,2374.0,1821.0,1542.0,1433.0,1597.0,1642.0,1826.0,2308.0,3012.0,2955.0,2380.0,1868.0,1584.0,1472.0,1604.0,1624.0,1801.0,2317.0,3013.0,2992.0,2368.0,1786.0,1498.0,1400.0,1566.0,1654.0,1824.0,2331.0,3015.0,3019.0,2708.0,2061.0,1590.0,1576.0,1723.0,1595.0
],[2091.0,2173.0,2244.0,2182.0,2097.0,2023.0,1942.0,1903.0,1742.0,1771.0,1977.0,2052.0,2093.0,2064.0,2052.0,2048.0,2000.0,1991.0,1823.0,1855.0,2100.0,2168.0,2184.0,2119.0,2075.0,2039.0,1992.0,1982.0,1816.0,1843.0,2015.0,2144.0,2232.0,2211.0,2181.0,2152.0,2116.0,2097.0,1939.0,1970.0,2200.0,2265.0,2246.0,2159.0,2092.0,2043.0,1999.0,1976.0,1816.0,1931.0,2120.0,2212.0,2244.0,2196.0,2142.0,2091.0,2030.0,2003.0,1841.0,1880.0,2095.0,2163.0,2195.0,2151.0,2112.0,2071.0,2016.0,2006.0,1830.0,2066.0,2149.0,2208.0,2223.0,2175.0,2121.0,2076.0,2018.0,2000.0,1835.0,1838.0,2112.0,2168.0,2168.0,2110.0,2064.0,2028.0,1991.0,2008.0,1882.0,1929.0,2196.0,2283.0,2342.0,2316.0,2264.0,2195.0,2115.0,2069.0,1877.0,1860.0
],[2116.0,2034.0,1978.0,1957.0,1886.0,1771.0,2050.0,2193.0,2244.0,2181.0,2116.0,2033.0,1984.0,1956.0,1882.0,1762.0,1968.0,2124.0,2259.0,2256.0,2180.0,2079.0,1993.0,1960.0,1872.0,1746.0,1876.0,2192.0,2270.0,2178.0,2098.0,2007.0,1956.0,1939.0,1868.0,1750.0,1880.0,2171.0,2284.0,2201.0,2114.0,2030.0,1974.0,1950.0,1885.0,1761.0,1928.0,2178.0,2292.0,2200.0,2119.0,2026.0,1968.0,1947.0,1879.0,1768.0,2095.0,2176.0,2253.0,2196.0,2034.0,2033.0,2032.0,2018.0,1932.0,1799.0,1934.0,2195.0,2308.0,2204.0,2104.0,2015.0,1956.0,1944.0,1865.0,1760.0,1971.0,2098.0,2284.0,2264.0,2192.0,2081.0,2006.0,1959.0,1879.0,1749.0,1882.0,2154.0,2272.0,2178.0,2082.0,1994.0,1937.0,1927.0,1856.0,1743.0,1878.0,2166.0,2298.0,2200.0
],[2800.0,3029.0,2564.0,2019.0,1615.0,1432.0,1479.0,1546.0,1765.0,2040.0,2796.0,3018.0,2550.0,2008.0,1600.0,1448.0,1502.0,1544.0,1762.0,2036.0,2900.0,3020.0,2996.0,2417.0,1808.0,1440.0,1345.0,1348.0,1557.0,1886.0,2656.0,2999.0,2540.0,2061.0,1652.0,1486.0,1528.0,1565.0,1778.0,2044.0,2793.0,3024.0,2552.0,1992.0,1610.0,1443.0,1505.0,1571.0,1779.0,2075.0,2808.0,3012.0,2520.0,1992.0,1636.0,1485.0,1532.0,1572.0,1808.0,2089.0,2832.0,3024.0,2511.0,1989.0,1600.0,1443.0,1477.0,1540.0,1759.0,2088.0,2828.0,3020.0,2504.0,1960.0,1592.0,1445.0,1516.0,1555.0,1766.0,2043.0,2785.0,3001.0,2859.0,2512.0,1900.0,1506.0,1384.0,1352.0,1560.0,1890.0,2679.0,3025.0,2468.0,1915.0,1552.0,1411.0,1520.0,1584.0,1802.0,2074.0
],[2012.0,1920.0,2003.0,2179.0,2222.0,2180.0,2142.0,2106.0,2060.0,2055.0,2005.0,1914.0,2000.0,2201.0,2240.0,2197.0,2145.0,2100.0,2055.0,2054.0,2002.0,1912.0,2039.0,2111.0,2215.0,2194.0,2160.0,2116.0,2064.0,2058.0,1999.0,1912.0,2056.0,2168.0,2196.0,2138.0,2107.0,2065.0,2024.0,2027.0,1991.0,1919.0,1993.0,2184.0,2220.0,2167.0,2127.0,2085.0,2047.0,2056.0,2008.0,1924.0,2024.0,2194.0,2223.0,2170.0,2131.0,2093.0,2048.0,2052.0,2015.0,1933.0,2027.0,2217.0,2249.0,2195.0,2150.0,2113.0,2062.0,2058.0,2019.0,1929.0,2012.0,2198.0,2252.0,2192.0,2144.0,2098.0,2052.0,2053.0,2006.0,1923.0,2107.0,2104.0,2216.0,2190.0,2104.0,2119.0,2087.0,2080.0,2024.0,1942.0,1999.0,2151.0,2228.0,2158.0,2112.0,2068.0,2029.0,2031.0
],[914.0,358.0,283.0,196.0,1096.0,1581.0,2050.0,2228.0,1836.0,2274.0,1696.0,2325.0,1933.0,1404.0,1005.0,713.0,481.0,883.0,1442.0,1574.0,1762.0,1866.0,2076.0,1816.0,2224.0,1776.0,1708.0,1496.0,1600.0,1376.0,960.0,682.0,809.0,1116.0,1195.0,1158.0,1081.0,1041.0,1024.0,1047.0,964.0,1145.0,1372.0,1500.0,1546.0,1565.0,1560.0,1583.0,1584.0,1620.0,1485.0,1650.0,1850.0,1959.0,1996.0,1980.0,1946.0,1938.0,2062.0,2003.0,1788.0,1764.0,1890.0,1959.0,1945.0,1897.0,1845.0,1848.0,1717.0,1740.0,1728.0,2273.0,2348.0,2424.0,2568.0,2597.0,2544.0,2498.0,2432.0,2388.0,2192.0,2310.0,2478.0,2534.0,2509.0,2433.0,2358.0,2306.0,2253.0,2228.0,2056.0,2176.0,2388.0,2452.0,2432.0,2353.0,2282.0,2234.0,2170.0,2148.0
]])

test_data = np.reshape(test_data, [(int)(len(test_data) / sensor_num), sensor_num, hz])

result = model.predict(test_data)
print(result)
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
