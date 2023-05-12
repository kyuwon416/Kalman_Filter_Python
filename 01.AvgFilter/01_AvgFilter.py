import numpy as np
import matplotlib.pyplot as plt

prevAvg = 0
k = 1

# Define Average Filter Function with Recursion
def AvgFilter(x):
    global prevAvg, k
    alpha = (k - 1) / k
    avg = alpha * prevAvg + (1 - alpha) * x
    prevAvg = avg
    k += 1
    return avg

# Define Measured Voltage
def GetVolt():
    w = np.random.normal(0, 4, 1)
    z = 14.4 + w
    return z

dt = 0.2
t = np.arange(0, 10, dt)

Nsamples = len(t)

Xmsaved = np.zeros(Nsamples)
Avgsaved = np.zeros(Nsamples)

for i in range(Nsamples):
    xm = GetVolt()
    avg = AvgFilter(xm)

    Xmsaved[i] = xm
    Avgsaved[i] = avg

plt.figure(figsize = (8, 5))
plt.plot(t, Xmsaved, 'b*--', label = 'Measured')
plt.plot(t, Avgsaved, 'ro-', label = 'Average')
plt.legend(loc = 'best')
plt.ylabel('Volt[V]')
plt.xlabel('Time[sec]')
plt.show()