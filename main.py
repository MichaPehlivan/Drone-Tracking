# External Imports
import numpy as np

# Internal Packages
from Track_Simulation import (
    simulateLinearTrack,
    simulateLinearTrackPolar,
    simulateRandomAccelTrackPolar,
    simulateRandomAccelHoverTrackPolar,
)
from Tracking_Routines import (
    RunSimpleKalman,
    RunExtendedKalman,
    RunUnscentedKalman,
    RunJointKalman,
)
from Utils import animate_track
from Utils.KalmanTuner import TuneEKF, TuneUKF

# EXTENDED KALMAN FILTER
# Initialize values
dt = 0.4
x_initial = 5
y_initial = 5
measurement_sigma = 0  # standard deviation of the measurement
var = measurement_sigma**2

num_datapoints = 60

# Initialize the functions and matrices for the Kalman filter.
f = lambda x: np.dot(
    np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]), x
)
F = lambda x: np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

h_cartesian = lambda x: np.array([x[0], x[1]])
H_cartesian = lambda x: np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
h_polar = lambda x: np.array(
    [[np.sqrt(x[0] ** 2 + x[1] ** 2)], [np.arctan2(x[1], x[0])]]
)  # conversion to polar
H_polar = lambda x: np.array(
    [
        [
            x[0] / np.sqrt(x[0] ** 2 + x[1] ** 2),
            x[1] / np.sqrt(x[0] ** 2 + x[1] ** 2),
            0,
            0,
        ],
        [
            (-1 * x[1]) / (x[0] ** 2 + x[1] ** 2),
            x[0] / (x[0] ** 2 + x[1] ** 2),
            0,
            0,
        ],  # derivative of arctan2
    ]
)

# TODO: Find appropriate covariance matrices.
Q = (
    100
    * 0.3**2
    * np.array(
        [
            [(dt**4) / 4, 0, (dt**3) / 2, 0],
            [0, (dt**4) / 4, 0, (dt**3) / 2],
            [(dt**3) / 2, 0, dt**2, 0],
            [0, (dt**3) / 2, 0, dt**2],
        ]
    )
)

R = 1 * np.array([[var, 0], [0, var]])

x0 = np.array([[x_initial], [y_initial], [1], [1]])

P0 = 0.0001 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# Run with no noise
range_sigma = 0
azimuth_sigma = np.deg2rad(0)
var_r = range_sigma**2
var_phi = azimuth_sigma**2
R = 1 * np.array([[var_r, 0], [0, var_phi]])
measurements, trueTrack = simulateRandomAccelTrackPolar(
    v_x=1,
    v_y=1,
    x0=x_initial,
    y0=y_initial,
    num_datapoints=num_datapoints,
    dt=dt,
    sigma_r=range_sigma,
    sigma_phi=azimuth_sigma,
)

# animate_track(trueTrack, dt=dt)
# RunJointKalman(
#     f, h_polar, F, H_polar, Q, R, x0, P0, 0.01, 2, 0, measurements, trueTrack
# )

# Another with low sigma
range_sigma = 0.5
azimuth_sigma = np.deg2rad(1)
var_r = range_sigma**2
var_phi = azimuth_sigma**2
R = 1 * np.array([[var_r, 0], [0, var_phi]])
measurements, trueTrack = simulateRandomAccelTrackPolar(
    v_x=1,
    v_y=1,
    x0=x_initial,
    y0=y_initial,
    num_datapoints=num_datapoints,
    dt=dt,
    sigma_r=range_sigma,
    sigma_phi=azimuth_sigma,
)

# animate_track(trueTrack, dt=dt)
RunJointKalman(f, h_polar, F, H_polar, Q, R, x0, P0, 2, 2, 0, measurements, trueTrack)
# TuneEKF(f, h_polar, F, H_polar, x0, var_r, var_phi, dt, measurements, trueTrack, 10)
# TuneUKF(f, h_polar, x0, var_r, var_phi, dt, 2, 0, measurements, trueTrack, 10)

# Another with high sigma
range_sigma = 10
azimuth_sigma = np.deg2rad(5)
var_r = range_sigma**2
var_phi = azimuth_sigma**2
R = 1 * np.array([[var_r, 0], [0, var_phi]])
measurements, trueTrack = simulateRandomAccelTrackPolar(
    v_x=1,
    v_y=1,
    x0=x_initial,
    y0=y_initial,
    num_datapoints=num_datapoints,
    dt=dt,
    sigma_r=range_sigma,
    sigma_phi=azimuth_sigma,
)

# animate_track(trueTrack, dt=dt)
# RunJointKalman(
#     f, h_polar, F, H_polar, Q, R, x0, P0, 0.01, 2, 0, measurements, trueTrack
# )
# TuneEKF(f, h_polar, F, H_polar, x0, var_r, var_phi, measurements, trueTrack)
# TuneUKF(f, h_polar, x0, var_r, var_phi, 2, 0, measurements, trueTrack)
