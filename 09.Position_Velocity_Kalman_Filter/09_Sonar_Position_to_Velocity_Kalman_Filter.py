'''
Date : 2023-05-13
E-mail : kyuwon416@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

# X is State Vector, P is State Covariance Matrix
# A is State Transition Matrix, H is State-To-Measurement Matrix
# Q is Process Noise Covariance Matrix, R is Measurement Covariance Matrix

def Kalman_Filter(X, P, A, H, Q, R, measured, previous_estimate, iteration):
    if iteration == 0:
        previous_estimate = X

    # Predict System State Estimate
    x_predict = A.dot(previous_estimate)
    p_predict = A.dot(P).dot(A.T) + Q

    # Compute the Kalman Gain
    kalman_gain = p_predict.dot(H.T).dot(np.linalg.inv(H.dot(p_predict).dot(H.T) + R))

    # Estimate System State & System State Covariance Matrix
    x_estimate = x_predict + kalman_gain.dot(measured - H.dot(x_predict))
    error_covariance = p_predict - kalman_gain.dot(H).dot(p_predict)

    return x_estimate, error_covariance, kalman_gain

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

Position_Measured_Saved = np.zeros(n_samples)
Position_Kalman_Saved = np.zeros(n_samples)
Velocity_Kalman_Saved = np.zeros(n_samples)
# P_Covariance_Saved = np.zeros(n_samples)
# K_Kalman_Gain_Saved = np.zeros(n_samples)

# Set Variable
# X is State Vector, P is State Covariance Matrix
# A is State Transition Matrix, H is State-To-Measurement Matrix
# Q is Process Noise Covariance Matrix, R is Measurement Covariance Matrix

# Initialize System State
X = np.array([0, 20]).transpose()
P = 5 * np.eye(2)

# System Model Variable
A = np.array([[1, dt],
              [0, 1]])
H = np.array([[1, 0]])
Q = np.array([[1, 0],
              [0, 3]])
R = np.array([[10]])

x_kalman = None

for i in range(n_samples):
    position_measured = Get_Sonar(i)
    position_measured = (position_measured - 40) / 2
    x_kalman, P, kalman_gain = Kalman_Filter(X, P, A, H, Q, R, position_measured, x_kalman, i)

    Position_Measured_Saved[i] = position_measured
    Position_Kalman_Saved[i] = x_kalman[0]
    Velocity_Kalman_Saved[i] = x_kalman[1]

fig, axes = plt.subplots(2, 1, figsize = (8, 10))

plt.subplot(2, 1, 1)
plt.plot(time, Position_Measured_Saved, 'b*--', markersize = 2, label='Measured Value')
plt.plot(time, Position_Kalman_Saved, 'ro-', markersize = 2, label='Kalman Filter')
plt.legend(loc = 'best')
plt.title('Kalman Filter : Position')
plt.xlabel('Time[sec]')
plt.ylabel('Position[m]')

plt.subplot(2, 1, 2)
plt.plot(time, Position_Kalman_Saved, 'b*--', markersize = 2, label='Distance')
plt.plot(time, Velocity_Kalman_Saved, 'ro-', markersize = 2, label='Velocity')
plt.legend(loc = 'best')
plt.title('Velocity & Distance')
plt.xlabel('Time[sec]')
plt.ylabel('Velocity[m/s]')
plt.show()
