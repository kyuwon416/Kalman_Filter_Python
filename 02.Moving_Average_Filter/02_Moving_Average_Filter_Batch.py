'''
Date : 2023-05-12
E-mail : kyuwon416@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

# Define Average Filter Function with Batch
def Moving_Average_Filter_Batch(n, measured, iteration):
    global x_n
    if iteration == 0:
        x_n = measured * np.ones(n)

    else:
        for i in range(n - 1):
            x_n[i] = x_n[i + 1]
        x_n[n - 1] = measured
    avg = np.mean(x_n)
    return avg

# Get_Sonar_Data
def Get_Sonar(x):
    sonar = data['sonarAlt'][0][x]
    return sonar

data = io.loadmat('./SonarAlt.mat')

n_samples = 500

time_start = 0
time_end = 10
dt = (time_end - time_start) / n_samples
time = np.arange(time_start, n_samples * dt, dt)

X_Measured_Saved = np.zeros(n_samples)
X_Moving_Average_Saved = np.zeros(n_samples)

n = int(input("Enter The Number of Moving Average Data : "))
x_n = []

for i in range(0, n_samples):
    x_measured = Get_Sonar(i)
    x_moving_average = Moving_Average_Filter_Batch(n, x_measured, i)

    X_Measured_Saved[i] = x_measured
    X_Moving_Average_Saved[i] = x_moving_average

plt.figure(figsize = (8, 5))
plt.plot(time, X_Measured_Saved, 'b*--', markersize = 2, label = 'Measured')
plt.plot(time, X_Moving_Average_Saved, 'ro-', markersize = 2, label = 'Moving Average')
plt.legend(loc = 'best')
plt.title('Moving Average(n={})'.format(n), fontsize = 15)
plt.ylabel('Altitude[m]')
plt.xlabel('Time[sec]')
plt.show()