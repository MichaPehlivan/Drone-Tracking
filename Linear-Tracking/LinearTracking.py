import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy.random import randn
from matplotlib.widgets import Slider
# this file will contain the code for the linear tracker.
# references: https://www.geeksforgeeks.org/python/kalman-filter-in-python/
#             https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
#             https://stackoverflow.com/questions/66007351/kalman-filter-2d-with-pykalman

#             https://aleksandarhaber.com/introduction-to-kalman-filter-derivation-of-the-recursive-least-squares-method-with-python-codes/
#             https://aleksandarhaber.com/time-propagation-of-state-vector-and-state-covariance-matrix-of-linear-dynamical-systems-intro-to-kalman-filtering/
#             https://aleksandarhaber.com/kalman-filter-complete-derivation-from-scratch/
'''
    Assumptions: 
    -Path is linear 
    -Use a normal Kalman filter
    -No clustering is applied yet

'''


class KalmanFilter:

    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x

def simulateMeasurements(v_x, v_y, x0, y0, num_datapoints, dt, sigma):
    """"
    velocities all in meter/s
    """
    x = x0
    y = y0
    t = 0
    measurements = np.zeros((2,num_datapoints))
    # print(measurements.shape)
    for i in range(num_datapoints):
        measurements[0,i] = x + v_x*t + sigma*randn()
        measurements[1,i] = y + v_y*t + sigma*randn()
        t += dt

    return measurements

if __name__ == '__main__':

    dt = 0.1 # timestep is 0.1 seconds
    sigma = 0.1 # standard deviation of the measurement noise
    '''
    state vector:
    x
    y
    x_dot (vx)
    y_dot (vy)
    '''

    # for measurement in measurements:
    num_datapoints = 10
    v_x = 10
    v_y = 10
    xposition0 = 0
    yposition0 = 0

    measurements = simulateMeasurements(v_x, v_y, xposition0, yposition0, num_datapoints, dt, sigma)

    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0,0, 0, 1] ])

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    Q = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]])

    R = np.array([[1, 0],
                  [0, 1]])

    x0 = np.array([[xposition0], [yposition0], [1], [1]])

    P0 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    KF = KalmanFilter(F, H, Q, R, x0, P0)
    x_history = np.zeros((4, num_datapoints))

    for i in range(num_datapoints):
        KF.predict()
        KF.update(measurements[:, i].reshape(2, 1))
        print(KF.x)
        x_history[:, i] = KF.x.reshape(4, )

    # Plot the raw measurements as distinct points (e.g., red 'x' marks)
    plt.scatter(measurements[0, :], measurements[1, :], color='red', marker='x', label='Measurements')

    # Plot the Kalman Filter estimates as a continuous tracking line (e.g., blue line)
    plt.plot(x_history[0, :], x_history[1, :], color='blue', label='Kalman Filter Track')

    # Add formatting to make the plot clear and readable
    plt.title('Kalman Filter Tracking')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)

    # Setting axis to 'equal' ensures the x and y scale are the same,
    # which is important for visualizing physical paths without distortion.
    plt.axis('equal')

    plt.show()

