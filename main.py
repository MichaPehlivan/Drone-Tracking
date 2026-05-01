import numpy as np

from Track_Simulation import simulateLinearTrack
from Tracking_Routines import RunSimpleKalman

# initialize values
dt = 0.1
x_initial = 0
y_initial = 0

# initialize the simulated measurements.
measurements = simulateLinearTrack(v_x=10, v_y = 10, x0 = x_initial, y0 = y_initial, num_datapoints = 15, dt = dt, sigma = 0)

# initialize the matrices for the Kalman filter.
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1,  0],
              [0, 0, 0,  1] ])

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# TODO: Find appropriate covariance matrices.
Q = np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]])

R = np.array([[1, 0],
              [0, 1]])

x0 = np.array([[x_initial],
               [y_initial],
                  [1],
                  [1]])

P0 = 1000*np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

# Run the kalman filter.
RunSimpleKalman(F, H, Q, R, x0, P0, measurements)
