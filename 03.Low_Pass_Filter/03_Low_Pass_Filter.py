'''
Date : 2023-05-12
E-mail : kyuwon416@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

# Define Low Pass Filter Function with Batch
def LowPassFilter_1(measured, previous_estimate, alpha, iteration):
    if iteration == 0:
        previous_estimate = measured

    x_estimate = alpha * previous_estimate + (1 - alpha) * measured
    return x_estimate

# Get Sonar Data
def Get_Sonar(x):
    sonar = data['sonarAlt'][0][x]
    return sonar

data = io.loadmat('./SonarAlt.mat')

n_samples = 500

time_start = 0
time_end = 10
dt = (time_end - time_start) / n_samples
time = np.arange(time_start, time_end, dt)

X_Measured_Saved = np.zeros(n_samples)
X_Low_Pass_Saved = np.zeros(n_samples)

alpha = float(input('Enter The Alpha of Low Pass Filter(0 < alpha < 1) : '))
x_low_pass = None

for i in range(0, n_samples):
    x_measured = Get_Sonar(i)
    x_low_pass = LowPassFilter_1(x_measured, x_low_pass, alpha, i)

    X_Measured_Saved[i] = x_measured
    X_Low_Pass_Saved[i] = x_low_pass

plt.figure(figsize = (8, 5))
plt.plot(time, X_Measured_Saved, 'b*--', markersize = 2, label = 'Measured')
plt.plot(time, X_Low_Pass_Saved, 'ro-', markersize = 2, label = 'LPF(alpha = {})'.format(alpha))
plt.legend(loc = 'best')
plt.title('Low Pass Filter')
plt.ylabel('Altitude[m]')
plt.xlabel('Time[sec]')
plt.show()