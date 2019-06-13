import numpy as np
import matplotlib.pylab as plt
import Data_load as dl
import seaborn as sns
import os
import random
from scipy.interpolate import spline
from scipy.signal import lfilter
import scipy.interpolate as spi
#from scipy.integrate import pchip

current_path = os.getcwd()

DataX = list()
DataY = list()

Data_Animal_X = list()
Data_Animal_Y = list()

Data_Both_X = list()
Data_Both_Y = list()

Data_sun_human = list()
Data_sun_human_Y = list()
Data_rain_human = list()
Data_rain_human_Y = list()
Data_sun_none = list()
Data_sun_none_Y = list()
Data_rain_none = list()
Data_rain_none_Y = list()

dl.Data_load("Sunny", "Animal", 3, Data_Animal_X, Data_Animal_Y)

dl.Data_load("Sunny", "Human", 3, DataX, DataY)

#dl.Data_load("Sunny", "Both", 3, Data_Both_X, Data_Both_Y)

dl.Data_load("Sunny", "Human", 3, Data_sun_human, Data_sun_human_Y)

dl.Data_load("Rain", "Human", 3, Data_rain_human, Data_rain_human_Y)

dl.Data_load("Sunny", "None", 3, Data_sun_none, Data_sun_none_Y)

dl.Data_load("Rain", "None", 3, Data_rain_none, Data_rain_none_Y)

'''

hum = list()
ani = list()
both = list()
non = list()

for j in range(100, 400, 20):
    hum.append(DataX[j])
    ani.append(Data_Animal_X[j])
    both.append(Data_Both_X[j])
    non.append(Data_None_X[j])
'''
#dl.Data_load("Sunny","Animal",2, DataX, DataY)
#dl.Data_load("Sunny","Human",2, DataX, DataY)
#dl.Data_load("Sunny","Both",2, DataX, DataY)
#dl.Data_load("Sunny","None",2, DataX, DataY)

#DataX = np.asarray(DataX)
#DataY = np.asarray(DataY)

sun = list()
rain = list()
sun_ani = list()
sun_none = list()
rain_none = list()

for i in range(0,3000):
    j = random.randrange(0, Data_sun_human.__len__())
    sun.append(Data_sun_human[j])
    j = random.randrange(0, Data_rain_human.__len__())
    rain.append(Data_rain_human[j])
    j = random.randrange(0, Data_Animal_X.__len__())
    sun_ani.append(Data_Animal_X[j])
    j = random.randrange(0, Data_sun_none.__len__())
    sun_none.append(Data_sun_none[j])
    j = random.randrange(0, Data_rain_none.__len__())
    rain_none.append(Data_rain_none[j])
    print(j)


'''
for j in range(10, 200, 7):
    sun.append(Data_sun_human[j])
    rain.append(Data_rain_human[j])
    sun_ani.append(Data_Animal_X[j])


n = 15
b = [1.0/n] * n
a = 1

def smooth(x):
    yy = lfilter(b,a,x)
    plt.plot(yy,linewidth=0.5, linestyle="-", c="b")

def smooth_red(x):
    yy = lfilter(b,a,x)
    plt.plot(yy,linewidth=0.5, linestyle="-", c="r")

for i in range(0, sun.__len__()):
    smooth(sun[i])
    smooth_red(rain[i])

plt.show()
'''


plt.title("Exist Human(Blue:Sun, Red:Rain)")
sns.tsplot(data = sun, color = "blue")
sns.tsplot(data = rain, color = "red")
plt.legend()
plt.show()

plt.clf()

#ax = sns.lineplot(x="Epoch", y="signal", data=sun, hue="Data")
#ax.show()

plt.title("Not Exist(Blue:Sun, Red:Rain)")
sns.tsplot(data = sun_none, color = "blue")
sns.tsplot(data = rain_none, color = "red")
plt.show()
plt.clf()

plt.title("Sunny Weather(Blue:Human, Red:Animal, Yellow:None")
sns.tsplot(data = sun, color = "blue")
sns.tsplot(data = sun_ani, color = "red")
sns.tsplot(data = sun_none, color="yellow")
plt.show()

#xnew = np.linspace(DataX[:10].min(), DataX[:10].max(), 300)

#power_smooth = spline()

