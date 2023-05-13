'''
Date : 2023-05-13
E-mail : kyuwon416@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.linalg import inv
from skimage.metrics import structural_similarity

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

# Get Ball Position
def get_ball_pos(iimg=0):
    """Return measured position of ball by comparing with background image file.
        - References:
        (1) Data Science School:
            https://datascienceschool.net/view-notebook/f9f8983941254a34bf0fee42c66c5539
        (2) Image Diff Calculation:
            https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python
    """
    # Read images.
    imageA = cv2.imread('./img/bg.jpg')
    imageB = cv2.imread('./img/{}.jpg'.format(iimg+1))

    # Convert the images to grayscale.
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two images,
    # ensuring that the difference image is returned.
    _, diff = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype('uint8')

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    xc = int(M['m10'] / M['m00'])  # center of x as true position.
    yc = int(M['m01'] / M['m00'])  # center of y as true position.

    v = np.random.normal(0, 15)  # v: measurement noise of position.

    xpos_meas = xc + v  # x_pos_meas: measured position in x (observable).
    ypos_meas = yc + v  # y_pos_meas: measured position in y (observable).

    return np.array([xpos_meas, ypos_meas])

n_samples = 24
dt = 1

X_Position_Measured_Saved = np.zeros(n_samples)
Y_Position_Measured_Saved = np.zeros(n_samples)
X_Position_Kalman_Saved_1 = np.zeros(n_samples)
Y_Position_Kalman_Saved_1 = np.zeros(n_samples)
X_Position_Kalman_Saved_2 = np.zeros(n_samples)
Y_Position_Kalman_Saved_2 = np.zeros(n_samples)

# Set Variable
# X is State Vector, P is State Covariance Matrix
# A is State Transition Matrix, H is State-To-Measurement Matrix
# Q is Process Noise Covariance Matrix, R is Measurement Covariance Matrix

# Initialize System State
X = np.array([0, 0, 0, 0])  # (x-pos, x-vel, y-pos, y-vel) by definition in book.
P1 = 100 * np.eye(4)
P2 = 100 * np.eye(4)

# System Model Variable
A = np.array([[ 1, dt,  0,  0],
              [ 0,  1,  0,  0],
              [ 0,  0,  1, dt],
              [ 0,  0,  0,  1]])
H = np.array([[ 1,  0,  0,  0],
              [ 0,  0,  1,  0]])

Q1 = 1.0 * np.eye(4)
Q2 = 0.01 * np.eye(4)

R = np.array([[50,  0],
              [ 0, 50]])

x_kalman_1 = None
x_kalman_2 = None

for i in range(n_samples):
    measured = get_ball_pos(i)
    x_kalman_1, P1, kalman_gain_1 = Kalman_Filter(X, P1, A, H, Q1, R, measured, x_kalman_1, i)
    x_kalman_2, P2, kalman_gain_2 = Kalman_Filter(X, P2, A, H, Q2, R, measured, x_kalman_2, i)

    X_Position_Measured_Saved[i] = measured[0]
    Y_Position_Measured_Saved[i] = measured[1]
    X_Position_Kalman_Saved_1[i] = x_kalman_1[0]
    Y_Position_Kalman_Saved_1[i] = x_kalman_1[2]
    X_Position_Kalman_Saved_2[i] = x_kalman_2[0]
    Y_Position_Kalman_Saved_2[i] = x_kalman_2[2]

fig = plt.figure(figsize=(8, 8))
plt.gca().invert_yaxis()
plt.scatter(X_Position_Measured_Saved, Y_Position_Measured_Saved, s=150, c="b", marker='*', label='Position: Measurements')
plt.scatter(X_Position_Kalman_Saved_1, Y_Position_Kalman_Saved_1, s=60, c="r", marker='o', label='Position: Kalman Filter(Q)')
plt.scatter(X_Position_Kalman_Saved_2, Y_Position_Kalman_Saved_2, s=60, c="g", marker='s', label='Position: Kalman Filter(Q/100)')
plt.legend(loc='upper left')
plt.title('Position : Estimation(Q) vs Estimation(Q/100)')
plt.xlabel('X-Position[m]')
plt.ylabel('Y-Position[m]')
plt.xlim((-10, 350))
plt.ylim((250, -10))
plt.show()

# plt.ion()
# for i in range(n_samples):
#     fig = plt.figure(figsize=(8, 8))
#     image = cv2.imread('./img/{}.jpg'.format(i+1))
#     imgplot = plt.imshow(image)
#     plt.scatter(X_Position_Measured_Saved[i], Y_Position_Measured_Saved[i], s=150, c="b", marker='*', label='Position : Measurements')
#     plt.scatter(X_Position_Kalman_Saved_1[i], Y_Position_Kalman_Saved_1[i], s=60, c="r", marker='o', label='Position : Kalman Filter(Q)')
#     plt.scatter(X_Position_Kalman_Saved_2[i], Y_Position_Kalman_Saved_2[i], s=60, c="g", marker='s', label='Position : Kalman Filter(Q/100)')
#     plt.legend(loc='upper left')
#     plt.title('Position: True vs Measurement vs Estimation(Q) vs Estimation(Q/100)')
#     plt.xlabel('X-Position[m]')
#     plt.ylabel('Y-Position[m]')
#     plt.xlim((-10, 350))
#     plt.ylim((250, -10))
#     fig.canvas.draw()
#     plt.pause(0.05)