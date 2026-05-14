# External Imports
import numpy as np
import matplotlib.pyplot as plt
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
from Utils.Wrapper_Functions import SimulatorPolar_stonesoup

from stonesoup.plotter import Plotter
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

# Initialize values
dt = 0.4
x_initial = 5
y_initial = 5
measurement_sigma = 0  # standard deviation of the measurement
var = measurement_sigma**2

num_datapoints = 60

# Initialize the functions and matrices for the Kalman filter.
f = lambda x: np.dot(
    np.array(
        [
            [1, 0, dt, 0, 0, 0],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 0, 1],
        ]
    ),
    x,
)

F = lambda x: np.array(
    [
        [1, 0, dt, 0, 0, 0],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, dt, 0, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 0, 1],
    ]
)

h_cartesian = lambda x: np.array([x[0], x[3]])
H_cartesian = lambda x: np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
h_polar = lambda x: np.array(
    [[np.sqrt(x[0] ** 2 + x[3] ** 2)], [np.arctan2(x[3], x[0])]]
)  # conversion to polar

H_polar = lambda x: np.array(
    [
        [
            x[0] / np.sqrt(x[0] ** 2 + x[3] ** 2),
            0,
            0,
            x[3] / np.sqrt(x[0] ** 2 + x[3] ** 2),
            0,
            0,
        ],
        [
            (-1 * x[3]) / (x[0] ** 2 + x[3] ** 2),
            0,
            0,
            x[0] / (x[0] ** 2 + x[3] ** 2),
            0,
            0,
        ],  # derivative of arctan2
    ]
)

Q = 0.1 * np.eye(6)

R = 1 * np.array([[var, 0], [0, var]])

x0 = np.array([[x_initial], [1], [1],[y_initial], [1], [1]])

P0 = 10 * np.eye(6)

range_sigma = 1
azimuth_sigma = np.deg2rad(3)
var_r = range_sigma**2
var_phi = azimuth_sigma**2
R = 1 * np.array([[var_r, 0], [0, var_phi]])

#Define measurement_model
measurement_model = CartesianToBearingRange(
        ndim_state=6,
        mapping=(0, 3),
        noise_covar=R
)

detections, ground_truth = SimulatorPolar_stonesoup(
    sim_function = simulateRandomAccelTrackPolar,
    measurement_model=measurement_model, v_x=1,
    v_y=1,
    x0=x_initial,
    y0=y_initial,
    num_datapoints=num_datapoints,
    dt=dt,
    sigma_r=range_sigma,
    sigma_phi=azimuth_sigma
)
































# fig = plt.figure()
# ax = fig.add_subplot(projection='polar')
# for detection_set in detections:
#     for det in detection_set:
#          ax.scatter(det.state_vector[0],det.state_vector[1], c="blue", alpha=1)
# plt.show()


# plotter = Plotter()
# plotter.plot_ground_truths(ground_truth, mapping=[0,3])
# plotter.plot_measurements(detections, measurement_model=measurement_model, mapping= [0,3])
# plt.grid()
# plt.show()
