import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

prevX1 = 0
firstRun1 = True
prevX2 = 0
firstRun2 = True

def LowPassFilter_1(x, alpha):
    global prevX1, firstRun1
    if firstRun1:
        prevX1 = x
        firstRun1 = False

    xlpf = alpha * prevX1 + (1 - alpha) * x
    prevX1 = xlpf
    return xlpf

def LowPassFilter_2(x, alpha):
    global prevX2, firstRun2
    if firstRun2:
        prevX2 = x
        firstRun2 = False

    xlpf = alpha * prevX2 + (1 - alpha) * x
    prevX2 = xlpf
    return xlpf

def GetSonar(x):
    z = data['sonarAlt'][0][x]
    return z

data = sp.io.loadmat('./SonarAlt.mat')

dt = 0.02
t = np.arange(0, Nsamples * dt, dt)

Nsamples = 500
Xsaved_1 = np.zeros(Nsamples)
Xsaved_2 = np.zeros(Nsamples)
Xmsaved = np.zeros(Nsamples)

for i in range(0, Nsamples):
    xm = GetSonar(i)
    x1 = LowPassFilter_1(xm, 0.8)
    x2 = LowPassFilter_2(xm, 0.9)

    Xsaved_1[i] = x1
    Xsaved_2[i] = x2
    Xmsaved[i] = xm

plt.figure(figsize = (8, 5))
plt.plot(t, Xmsaved, 'b*--', label = 'Measured')
plt.plot(t, Xsaved_1, 'r-', label = 'LPF(alpha = 0.5)')
plt.legend(loc = 'best')
plt.ylabel('Altitude[m]')
plt.xlabel('Time[sec]')
plt.show()