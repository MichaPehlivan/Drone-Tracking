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


class ExtendedKalmanFilter:
    """
    Extended Kalman filter class.

    inputs:
        f: State transition function
        h: Observation function
        F: Jacobian of f
        H: Jacobian of h
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

    def __init__(self, f, h, F, H, Q, R, x0, P0):
        self.f = f
        self.h = h
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self):
        A = self.F(self.x)
        self.x = self.f(self.x).flatten()
        self.P = np.dot(A, np.dot(self.P, A.T)) + self.Q
        return self.x

    def update(self, z):
        A = self.H(self.x)
        z = z.flatten()
        S = np.dot(A, np.dot(self.P, A.T)) + self.R
        K = np.dot(np.dot(self.P, A.T), np.linalg.inv(S))
        y = z - self.h(self.x).flatten()
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, A), self.P)
        return self.x


class UnscentedKalmanFilter:
    """
    Unscented Kalman filter class.

    inputs:
        f: State transition function
        h: Observation function
        R: Measurement noise covariance TODO: figure out appropriate values 1
        x0: Initial state estimate
            state vector:
                |    x         |
                |    y         |
                |  x_dot (vx)  |
                |_ y_dot (vy) _|
        Q: Process noise covariance (uncertainty in the process)  TODO: figure out appropriate values 2
        P0: Initial Error covariance (needs to be a large value)
        alpha: scaling parameter
        beta: scaling parameter
        kappa: scaling parameter

    Methods:
        predict: predicts a new state based on current state.

        update: updates the estimate using the measurement and the kalman gain.
            z: Measurement (cartesian coordinates)
    """

    def __init__(self, f, h, Q, R, x0, P0, alpha, beta, kappa):
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.L = len(x0)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.labda = alpha**2 * (len(x0) + kappa) - len(x0)
        self.wm, self.wc = self.compute_weights()

    def compute_weights(self):
        wm = np.full(2 * self.L + 1, 1 / (2 * (self.L + self.labda)))
        wc = np.full(2 * self.L + 1, 1 / (2 * (self.L + self.labda)))
        wm[0] = self.labda / (self.L + self.labda)
        wc[0] = wm[0] + (1 - self.alpha**2 + self.beta)

        return wm, wc

    def generate_sigma_points(self, x, P):
        x = x.flatten()
        sigma_points = np.zeros((2 * self.L + 1, self.L))
        chol = np.linalg.cholesky((self.L + self.labda) * P)

        sigma_points[0] = x
        for i in range(self.L):
            sigma_points[i + 1] = x + chol[:, i]
            sigma_points[self.L + i + 1] = x - chol[:, i]
        return sigma_points

    def predict(self):
        # generate sigma points for current state
        sigma_points = self.generate_sigma_points(self.x, self.P)
        transformed_sigma_points = np.array([self.f(s).flatten() for s in sigma_points])

        self.x = np.dot(self.wm, transformed_sigma_points)
        self.P = np.copy(self.Q)
        for i in range(2 * self.L + 1):
            d = transformed_sigma_points[i] - self.x
            self.P += self.wc[i] * np.outer(d, d)

        return self.x, transformed_sigma_points

    def update(self, transformed_sigma_points, z):
        z = z.flatten()
        transformed_sigma_points = self.generate_sigma_points(self.x, self.P)
        y = np.array([self.h(s).flatten() for s in transformed_sigma_points])
        # Instead of simple dot product for the whole vector:
        y_mean = np.zeros(2)
        y_mean[0] = np.dot(self.wm, y[:, 0])  # Range is linear, dot product is fine

        # For the Azimuth (Angle):
        sum_sin = np.sum(self.wm * np.sin(y[:, 1]))
        sum_cos = np.sum(self.wm * np.cos(y[:, 1]))
        y_mean[1] = np.arctan2(sum_sin, sum_cos)
        y_cov = np.copy(self.R)
        cross_cov = np.zeros((self.L, y_mean.shape[0]))
        for i in range(2 * self.L + 1):
            d = y[i] - y_mean
            d[1] = (d[1] + np.pi) % (2 * np.pi) - np.pi
            d2 = transformed_sigma_points[i] - self.x
            y_cov += self.wc[i] * np.outer(d, d)
            cross_cov += self.wc[i] * np.outer(d2, d)
        K = np.dot(cross_cov, np.linalg.inv(y_cov))
        diff = z - y_mean
        diff[1] = (diff[1] + np.pi) % (2 * np.pi) - np.pi
        self.x = self.x + np.dot(K, diff)
        self.P = self.P - np.dot(np.dot(K, y_cov), K.T)
        return self.x
