import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

prevAvg = 0
n = 0
k = 0
xbuf = []
firstRun = True

def MovAvgFilter_Recur(x):
    global prevAvg, n, k, xbuf, firstRun
    if firstRun:
        n = 10
        xbuf = x * np.ones(n)
        k = 1
        prevAvg = x
        firstRun = False

    else:
        for i in range(n - 1):
            xbuf[i] = xbuf[i + 1]
        xbuf[n - 1] = x

    avg = np.sum(xbuf) / n
    prevAvg = avg
    return avg

def MovAvgFilter_Batch(x):
    global n, xbuf, firstRun
    if firstRun:
        n = 10
        xbuf = x * np.ones(n)
        firstRun = False

    else:
        for i in range(n - 1):
            xbuf[i] = xbuf[i + 1]
        xbuf[n - 1] = x
    avg = np.sum(xbuf) / n
    return avg

def GetSonar(x):
    z = data['sonarAlt'][0][x]
    return z

data = sp.io.loadmat('./SonarAlt.mat')

dt = 0.02
t = np.arange(0, NSamples * dt, dt)

NSamples = 500
Xsaved = np.zeros(NSamples)
Xmsaved = np.zeros(NSamples)

for i in range(0, NSamples):
    xm = GetSonar(i)
    x = MovAvgFilter_Batch(xm)

    Xsaved[i] = x
    Xmsaved[i] = xm

plt.figure(figsize = (8, 5))
plt.plot(t, Xmsaved, 'b*--', label = 'Measured')
plt.plot(t, Xsaved, 'ro-', label = 'Moving Average')
plt.legend(loc = 'best')
plt.ylabel('Altitude[m]')
plt.xlabel('Time[sec]')
plt.show()