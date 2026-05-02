# External Imports
import numpy as np

# Internal Packages
from Track_Simulation import simulateLinearTrack
from Tracking_Routines import RunSimpleKalman

# Initialize values
dt = 0.1
x_initial = 0
y_initial = 0
measurement_sigma = 0. # standard deviation of the measurement
var = measurement_sigma ** 2

num_datapoints = 30
# Initialize the simulated measurements.
measurements = simulateLinearTrack(v_x=1, v_y = 1, x0 = x_initial, y0 = y_initial, num_datapoints = num_datapoints, dt = dt, sigma = measurement_sigma)

# Initialize the matrices for the Kalman filter.
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1,  0],
              [0, 0, 0,  1] ])

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# TODO: Find appropriate covariance matrices.
Q = 0.0001* np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]])

R = 1*np.array([[var,  0],
              [0,  var]])

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
