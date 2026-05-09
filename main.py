# External Imports
import numpy as np

# Internal Packages
from Track_Simulation import simulateLinearTrack, simulateLinearTrackPolar, simulateRandomAccelTrackPolar, simulateRandomAccelHoverTrackPolar
from Tracking_Routines import RunSimpleKalman, RunExtendedKalman
from Utils import animate_track
# # NORMAL KALMAN FILTER
# # Initialize values
# dt = 0.1
# x_initial = 0
# y_initial = 0
# measurement_sigma = 0  # standard deviation of the measurement
# var = measurement_sigma**2

# num_datapoints = 10
# # Initialize the simulated measurements.
# measurements, trueTrack = simulateLinearTrack(
#     v_x=10,
#     v_y=10,
#     x0=x_initial,
#     y0=y_initial,
#     num_datapoints=num_datapoints,
#     dt=dt,
#     sigma=measurement_sigma,
# )

# # Initialize the matrices for the Kalman filter.
# F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

# H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

# # TODO: Find appropriate covariance matrices.
# Q = 0.0001 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# R = 1 * np.array([[var, 0], [0, var]])

# x0 = np.array([[x_initial], [y_initial], [1], [1]])

# P0 = 1000 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# # Run the kalman filter.
# RunSimpleKalman(F, H, Q, R, x0, P0, measurements, trueTrack)


# # Run another with high sigma (high ospa distance)
# measurement_sigma = 0.5
# var = measurement_sigma**2
# R = np.array([[var, 0], [0, var]])
# measurements, trueTrack = simulateLinearTrack(
#     v_x=10,
#     v_y=10,
#     x0=x_initial,
#     y0=y_initial,
#     num_datapoints=num_datapoints,
#     dt=dt,
#     sigma=measurement_sigma,
# )

# RunSimpleKalman(F, H, Q, R, x0, P0, measurements, trueTrack)


# # Another with zero sigma (zero ospa distance)
# measurement_sigma = 5
# var = measurement_sigma**2
# R = np.array([[var, 0], [0, var]])
# measurements, trueTrack = simulateLinearTrack(
#     v_x=10,
#     v_y=10,
#     x0=x_initial,
#     y0=y_initial,
#     num_datapoints=num_datapoints,
#     dt=dt,
#     sigma=measurement_sigma,
# )

# RunSimpleKalman(F, H, Q, R, x0, P0, measurements, trueTrack)

# EXTENDED KALMAN FILTER
# Initialize values
dt = 0.4
x_initial = 5
y_initial = 5
measurement_sigma = 0  # standard deviation of the measurement
var = measurement_sigma**2

num_datapoints = 60
# Initialize the simulated measurements.
measurements_cartesian, trueTrack = simulateLinearTrack(
    v_x=10,
    v_y=10,
    x0=x_initial,
    y0=y_initial,
    num_datapoints=num_datapoints,
    dt=dt,
    sigma=measurement_sigma,
)

measurements_polar, trueTrack = simulateRandomAccelTrackPolar(
    v_x=10,
    v_y=10,
    x0=x_initial,
    y0=y_initial,
    num_datapoints=num_datapoints,
    dt=dt,
    sigma_r=measurement_sigma,
    sigma_phi=measurement_sigma,
)

# Initialize the functions and matrices for the Kalman filter.
f = lambda x: np.dot(
    np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]), x
)
F = lambda x: np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

h_cartesian = lambda x: np.array([x[0], x[1]])
H_cartesian = lambda x: np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
h_polar = lambda x: np.array(
    [[np.sqrt(x[0][0] ** 2 + x[1][0] ** 2)], [np.arctan2(x[1][0], x[0][0])]]
)  # conversion to polar
H_polar = lambda x: np.array(
    [
        [
            x[0][0] / np.sqrt(x[0][0] ** 2 + x[1][0] ** 2),
            x[1][0] / np.sqrt(x[0][0] ** 2 + x[1][0] ** 2),
            0,
            0,
        ],
        [
            (-1 * x[1][0]) / (x[0][0] ** 2 + x[1][0] ** 2),
            x[0][0] / (x[0][0] ** 2 + x[1][0] ** 2),
            0,
            0,
        ],  # derivative of arctan2
    ]
)

# TODO: Find appropriate covariance matrices.
Q = 0.3 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

R = 1 * np.array([[var, 0], [0, var]])

x0 = np.array([[x_initial], [y_initial], [1], [1]])

P0 = 1000 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# Run the kalman filter with no noise.
# RunExtendedKalman(
#     f,
#     h_cartesian,
#     F,
#     H_cartesian,
#     Q,
#     R,
#     x0,
#     P0,
#     measurements_cartesian,
#     trueTrack,
#     polar=False,
# )  # cartesian

# RunExtendedKalman(
#     f, h_polar, F, H_polar, Q, R, x0, P0, measurements_polar, trueTrack
# )  # polar
#
# # Run another with low sigma
# range_sigma = 0.5
# azimuth_sigma = 0.02  # 1.14 degrees
# var_r = range_sigma**2
# var_phi = azimuth_sigma**2
# R = np.array([[var_r, 0], [0, var_phi]])
# measurements, trueTrack = simulateLinearTrackPolar(
#     v_x=10,
#     v_y=10,
#     x0=x_initial,
#     y0=y_initial,
#     num_datapoints=num_datapoints,
#     dt=dt,
#     sigma_r=range_sigma,
#     sigma_phi=azimuth_sigma,
# )

# RunExtendedKalman(f, h_polar, F, H_polar, Q, R, x0, P0, measurements_polar, trueTrack)
# np.random.seed(3000)

# Another with high sigma
range_sigma = 0.1
azimuth_sigma = np.deg2rad(5)  # 11.4 degrees
var_r = range_sigma**2
var_phi = azimuth_sigma**2
R = np.array([[var_r, 0],
              [0, var_phi]])
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

animate_track(trueTrack, dt=dt)
RunExtendedKalman(f, h_polar, F, H_polar, Q, R, x0, P0, measurements, trueTrack)

