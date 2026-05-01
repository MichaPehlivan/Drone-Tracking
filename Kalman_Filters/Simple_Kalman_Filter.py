# External Import
import numpy as np

"""
This file contains the class for the simple kalman filter implementation.
References used are contained in the README.md file.
"""

class KalmanFilter:
    """
    Simple Kalman filter class.

    inputs:
        F: State transition matrix (system model)
        H: Observation matrix
        R: Measurement noise covariance TODO: figure out appropriate values 1
        x0: Initial state estimate
            state vector:
                |    x         |
                |    y         |
                |  x_dot (vx)  |
                |_ y_dot (vy) _|
        Q: Process noise covariance (uncertainty in the process)  TODO: figure out appropriate values 2
        P0: Initial Error covariance (needs to be a large value)

    Methods:
        predict: predicts a new state based on current state.

        update: updates the estimate using the measurement and the kalman gain.
            z: Measurement (cartesian coordinates)
    """

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
