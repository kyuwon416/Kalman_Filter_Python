'''
Date : 2023-05-12
E-mail : kyuwon416@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt

# Define Average Filter Function with Recursion
def Average_Filter(measured, previous_average, iteration):
    if iteration == 0:
        avg = measured

    else:
        alpha = ((iteration + 1) - 1) / (iteration + 1)
        avg = alpha * previous_average + (1 - alpha) * measured

    return avg

# Define Measured Voltage Value
def Get_Volt():
    noise = np.random.normal(0, 4, 1)
    true_value = 14.4
    measured_value = true_value + noise

    return measured_value

time_start = 0
time_end = 10
dt = 0.2
time = np.arange(time_start, time_end, dt)

n_samples = len(time)

X_Measured_Saved = np.zeros(n_samples)
X_Avg_Saved = np.zeros(n_samples)

x_average = None

for i in range(n_samples):
    x_measured = Get_Volt()
    x_average = Average_Filter(x_measured, x_average, i)

    X_Measured_Saved[i] = x_measured
    X_Avg_Saved[i] = x_average

plt.figure(figsize = (8, 5))
plt.plot(time, X_Measured_Saved, 'b*--', markersize = 5, label = 'Measured')
plt.plot(time, X_Avg_Saved, 'ro-', markersize = 5, label = 'Average')
plt.legend(loc = 'best')
plt.title('Average Filter', fontsize = 15)
plt.ylabel('Volt[V]')
plt.xlabel('Time[sec]')
plt.show()