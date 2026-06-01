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


class UCMKalmanFilter:
    """
    Unbiased Converted Measurement Kalman filter class.

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

    def __init__(self, F, H, Q, P0, sigma_r, sigma_phi, x0):
        self.F = F
        self.H = H
        self.Q = Q
        self.P = P0
        self.sigma_r = sigma_r
        self.labda_phi = np.exp(-1 * (sigma_phi**2) / 2)
        self.x = x0

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        z = z.flatten()

        # compute R
        rxx = (self.labda_phi ** (-2) - 2) * z[1] ** 2 * np.cos(z[0]) ** 2 + 0.5 * (
            z[1] ** 2 + self.sigma_r**2
        ) * (1 + self.labda_phi**4 * np.cos(2 * z[0]))
        ryy = (self.labda_phi ** (-2) - 2) * z[1] ** 2 * np.sin(z[0]) ** 2 + 0.5 * (
            z[1] ** 2 + self.sigma_r**2
        ) * (1 - self.labda_phi**4 * np.cos(2 * z[0]))
        rxy = (self.labda_phi ** (-2) - 2) * z[1] ** 2 * np.cos(z[0]) * np.sin(
            z[0]
        ) + 0.5 * (z[1] ** 2 + self.sigma_r**2) * self.labda_phi**4 * np.sin(
            2 * z[0]
        )
        self.R = np.array([[rxx, rxy], [rxy, ryy]])

        # convert z
        z_converted = np.zeros((2, 1))
        z_converted[0, 0] = self.labda_phi**-1 * z[1] * np.cos(z[0])
        z_converted[1, 0] = self.labda_phi**-1 * z[1] * np.sin(z[0])

        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z_converted - np.dot(self.H, self.x)
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

    def __init__(self, f, h, F, H, Q, R, P0, x0):
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

    def __init__(self, f, h, Q, R, P0, alpha, beta, kappa, x0):
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.P = P0
        self.x = x0.flatten()
        self.L = len(x0)

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.labda = alpha**2 * (len(x0) + kappa) - len(x0)

        self.wm, self.wc = self.compute_weights()

    # calculate mean while normalizing angle
    def calculate_polar_mean(self, y):
        y_mean = np.zeros(2)
        y_mean[1] = np.dot(self.wm, y[:, 1])

        sum_sin = np.sum(self.wm * np.sin(y[:, 0]))
        sum_cos = np.sum(self.wm * np.cos(y[:, 0]))
        y_mean[0] = np.arctan2(sum_sin, sum_cos)

        return y_mean

    # keep angle in [-pi, pi]
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # compute sigma point weights
    def compute_weights(self):
        wm = np.full(2 * self.L + 1, 1 / (2 * (self.L + self.labda)))
        wc = np.full(2 * self.L + 1, 1 / (2 * (self.L + self.labda)))
        wm[0] = self.labda / (self.L + self.labda)
        wc[0] = wm[0] + (1 - self.alpha**2 + self.beta)

        return wm, wc

    def generate_sigma_points(self, x, P):
        sigma_points = np.zeros((2 * self.L + 1, self.L))

        try:
            chol = np.linalg.cholesky((self.L + self.labda) * P)
        except (
            np.linalg.LinAlgError
        ):  # if matrix is not positive semidefinite, try to make it so
            eps = 1e-9 * np.eye(self.L)
            chol = np.linalg.cholesky((self.L + self.labda) * (P + eps))

        sigma_points[0] = x
        for i in range(self.L):
            sigma_points[i + 1] = x + chol[:, i]
            sigma_points[self.L + i + 1] = x - chol[:, i]

        return sigma_points

    def predict(self):
        # generate sigma points from current estimate
        sigma_points = self.generate_sigma_points(self.x, self.P)
        # pass sigma points trough system model
        transformed_sigma_points = np.array([self.f(s).flatten() for s in sigma_points])

        # estimate mean by summing sigma points with weights
        self.x = np.dot(self.wm, transformed_sigma_points)

        # estimate covariance
        dx = transformed_sigma_points - self.x
        self.P = np.dot((self.wc * dx.T), dx) + self.Q

        return self.x

    def update(self, z):
        z = z.flatten()

        # recompute sigma points from most recent estimate
        transformed_sigma_points = self.generate_sigma_points(self.x, self.P)
        # pass sigma points trough observation model
        y = np.array([self.h(s).flatten() for s in transformed_sigma_points])

        # estimate mean while keeping angles intact
        y_mean = self.calculate_polar_mean(y)

        # compute residuals
        dy = y - y_mean
        dy[:, 0] = self.normalize_angle(dy[:, 0])
        dx = transformed_sigma_points - self.x

        # estimate covariances
        Pyy = np.dot((self.wc * dy.T), dy) + self.R
        Pxy = np.dot((self.wc * dx.T), dy)

        # kalman gain
        K = np.dot(Pxy, np.linalg.inv(Pyy))

        S = z - y_mean
        S[0] = self.normalize_angle(S[0])

        self.x = self.x + np.dot(K, S)
        self.P = self.P - np.dot(np.dot(K, Pyy), K.T)

        return self.x
