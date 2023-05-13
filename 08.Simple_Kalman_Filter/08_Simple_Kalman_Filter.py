'''
Date : 2023-05-13
E-mail : kyuwon416@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt

# Set Variable
# X is State Vector, P is State Covariance Matrix
# A is State Transition Matrix, H is State-To-Measurement Matrix
# Q is Process Noise Covariance Matrix, R is Measurement Covariance Matrix

def Kalman_Filter_One_Variable(X, P, A, H, Q, R, measured, previous_estimate, iteration):
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
X_Kalman_Saved = np.zeros(n_samples)
P_Covariance_Saved = np.zeros(n_samples)
K_Kalman_Gain_Saved = np.zeros(n_samples)

# Set Variable
# X is State Vector, P is State Covariance Matrix
# A is State Transition Matrix, H is State-To-Measurement Matrix
# Q is Process Noise Covariance Matrix, R is Measurement Covariance Matrix

# Initialize System State
X = 14
P = np.array([[6]])

# System Model Variable
A = np.array([[1]])
H = np.array([[1]])
Q = np.array([[0]])
R = np.array([[4]])

x_kalman = None

for i in range(n_samples):
    x_measured = Get_Volt()
    x_kalman, P, kalman_gain = Kalman_Filter_One_Variable(X, P, A, H, Q, R, x_measured, x_kalman, i)
    X_Measured_Saved[i] = x_measured
    X_Kalman_Saved[i] = x_kalman
    P_Covariance_Saved[i] = P
    K_Kalman_Gain_Saved[i] = kalman_gain

fig, axes = plt.subplots(3, 1, figsize = (8, 15))

plt.subplot(3, 1, 1)
plt.plot(time, X_Measured_Saved, 'b*--', markersize = 5, label = 'Measured')
plt.plot(time, X_Kalman_Saved, 'ro-', markersize = 5, label = 'Kalman Filter')
plt.legend(loc = 'best')
plt.title('Kalman Filter')
plt.ylabel('Altitude[m]')
plt.xlabel('Time[sec]')

plt.subplot(3, 1, 2)
plt.plot(time, P_Covariance_Saved, 'ro--', markersize = 5)
plt.title('Error Covariance')
plt.ylabel('Error Covariance')
plt.xlabel('Time[sec]')

plt.subplot(3, 1, 3)
plt.plot(time, K_Kalman_Gain_Saved, 'ro--', markersize = 5)
plt.title('Kalman Gain')
plt.ylabel('Kalman Gain')
plt.xlabel('Time[sec]')
plt.show()