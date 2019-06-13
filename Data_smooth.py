import numpy
import Data_load as dl

def smooth(x, window_len=100, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


from numpy import *
from pylab import *
import os
import random

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

sun = list()
rain = list()
sun_ani = list()
sun_none = list()
rain_none = list()

for i in range(0,10):
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

def smooth_data(datalist, plot_title):

    for i in range(0,10):
        t = linspace(-4, 4, 100)
        x = datalist[i]
        xn = x + randn(len(t)) * 0.1
        y = smooth(x)

        ws = 31

        subplot(211)

        plot(x)

        windows = ['flat']

        title(plot_title)
        subplot(212)

        plot(smooth(xn, 10, 'flat'))

        title("Smoothing")

    show()

smooth_data(sun,"Sunny & Human")
smooth_data(rain, "Rain & Human")

smooth_data(sun_none, "Sunny & None")
smooth_data(rain_none, "Rain & None")

smooth_data(sun_ani, "Sunny & Animal")