import numpy as np
import matplotlib.pylab as plt
import Data_load_4 as dl
import seaborn as sns
import os
import random
from scipy.interpolate import spline
from scipy.signal import lfilter
import scipy.interpolate as spi
#from scipy.integrate import pchip

current_path = os.getcwd()

DataX_human = list()
DataY_human = list()

current_path = os.getcwd()

DataX_none = list()
DataY_none = list()

## Data 순서 :: pir , light, temp , humid
dl.Data_load("None", DataX_human, DataY_human)

dl.Data_load("Human", DataX_none, DataY_none)

human = list()
none = list()

DataX_human = np.asarray(DataX_human)
DataX_human = np.reshape(DataX_human, (-1 , 1 , 100))

DataX_none = np.asarray(DataX_none)
DataX_none = np.reshape(DataX_none, (-1, 1, 100))
'''
for i in range(0,2000):
    j = random.randrange(0, DataX_human.__len__())
    human.append(DataX_human[j][0])
    j = random.randrange(0, DataX_none.__len__())
    none.append(DataX_none[j][0])
'''

for i in range(0,2000):
    j = random.randrange(0, DataX_human.__len__())
    human.append(DataX_human[j][0])
    j = random.randrange(0, DataX_none.__len__())
    none.append(DataX_none[j][0])

plt.title("Blue : Human, Red : Not exist")

sns.tsplot(data=human, color="blue")
sns.tsplot(data=none, color="red")

plt.show()
'''
plt.plot(human, color="blue")
plt.plot(none, color="red")
plt.show()
plt.clf()
'''
'''
plt.title("Blue : Human, Red : Not exist")
plt.plot(human, color = "blue")
plt.plot(none, color = "red")
plt.show()
plt.clf()
'''
'''
plt.title("10 data .. Human")
plt.plot(human[0], color = "blue")
plt.plot(human[1], color = "blue")
plt.plot(human[2], color = "blue")
plt.plot(human[3], color = "blue")
plt.plot(human[4], color = "blue")
plt.plot(human[5], color = "blue")
plt.plot(human[6], color = "blue")
plt.plot(human[7], color = "blue")
plt.plot(human[8], color = "blue")
plt.plot(human[9], color = "blue")
plt.show()

plt.clf()

plt.title("10 data .. not exist")

plt.plot(none[0], color = "red")
plt.plot(none[1], color = "red")
plt.plot(none[2], color = "red")
plt.plot(none[3], color = "red")
plt.plot(none[4], color = "red")
plt.plot(none[5], color = "red")
plt.plot(none[6], color = "red")
plt.plot(none[7], color = "red")
plt.plot(none[8], color = "red")
plt.plot(none[9], color = "red")

plt.show()

plt.clf()
'''