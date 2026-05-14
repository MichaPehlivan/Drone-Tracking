# External Imports
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Internal Packages
from Evaluation_Metrics import average_ospa_stonesoup
from Kalman_Filters import UnscentedKalmanFilter
from Utils.KalmanTuner import TuneEKF, TuneUKF
from Track_Simulation import (
    simulateLinearTrack,
    simulateLinearTrackPolar,
    simulateRandomAccelTrackPolar,
    simulateRandomAccelHoverTrackPolar,
)
from Utils.Wrapper_Functions import (
    SimulatorPolar_stonesoup,
    UKFUpdater,
    UKFPredictor
)

#Stonesoup imports
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

from stonesoup.plotter import Plotter



# Initialize values
dt = 0.4
x_initial = 5
y_initial = 5

num_datapoints = 50

# Initialize the functions and matrices for the Kalman filter.
f = lambda x: np.dot(np.array([
    [1, dt, 0.5*dt**2, 0, 0,  0        ],  # x
    [0, 1,  dt,        0, 0,  0        ],  # vx
    [0, 0,  1,         0, 0,  0        ],  # ax
    [0, 0,  0,         1, dt, 0.5*dt**2],  # y
    [0, 0,  0,         0, 1,  dt       ],  # vy
    [0, 0,  0,         0, 0,  1        ],  # ay
]), x)

F = lambda x: np.array([
    [1, dt, 0.5*dt**2, 0, 0,  0],          # x
    [0,  1,     dt,    0, 0,  0],          # vx
    [0,  0,     1,     0, 0,  0],          # ax
    [0,  0,     0,     1, dt, 0.5*dt**2],  # y
    [0,  0,     0,     0, 1,  dt],         # vy
    [0,  0,     0,     0, 0,  1],          # ay
])

h_cartesian = lambda x: np.array([x[0], x[3]])

H_cartesian = lambda x: np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

h_polar = lambda x: np.array([np.arctan2(x[3], x[0]),
                              np.sqrt(x[0]**2 + x[3]**2)]
                             )

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
# Qblock = 0.1 * np.array([
#     [dt**5/20, dt**4/8, dt**3/6],
#     [dt**4/8,  dt**3/3, dt**2/2],
#     [dt**3/6,  dt**2/2, dt     ],
# ])
#
# Q = np.zeros((6, 6))
# Q[0:3, 0:3] = Qblock
# Q[3:6, 3:6] = Qblock

np.random.seed(30)

x0 = np.array([[x_initial], [1], [0], [y_initial], [1], [0]])

P0 = 1 * np.eye(6)

range_sigma = 1
azimuth_sigma = np.deg2rad(5)
var_r = range_sigma**2
var_phi = azimuth_sigma**2

R = 1 * np.array([[var_phi, 0], [0, var_r]])

#Define measurement_model
measurement_model = CartesianToBearingRange(
        ndim_state=6,
        mapping=(0, 3),
        noise_covar=R
)

timestamp = datetime.datetime.now()

detections, ground_truth = SimulatorPolar_stonesoup(
    sim_function = simulateRandomAccelTrackPolar,
    start_time = timestamp,
    measurement_model=measurement_model, v_x=1,
    v_y=1,
    x0=x_initial,
    y0=y_initial,
    num_datapoints=num_datapoints,
    dt=dt,
    sigma_r=range_sigma,
    sigma_phi=azimuth_sigma
)

ukf = UnscentedKalmanFilter(
    f=f,
    h=h_polar,
    Q=Q,
    R=R,
    x0=x0,
    P0=P0,
    alpha=0.3,
    beta=2,
    kappa=0,
)


predictor = UKFPredictor(ukf=ukf)
updater = UKFUpdater(ukf=ukf)

timestamp = datetime.datetime.now()

prior = GaussianState(
    state_vector=x0,
    covar=ukf.P,
    timestamp=timestamp,
)

track = Track()

for detection_set in detections:
    for detection in detection_set:

        timestamp += datetime.timedelta(seconds=dt)

        prediction = predictor.predict(prior, timestamp=timestamp)


        hypothesis = SingleHypothesis(prediction=prediction, measurement=detection)
        posterior = updater.update(hypothesis)

        track.append(posterior)

        prior = posterior

avg_ospa = average_ospa_stonesoup(track = track, ground_truth=ground_truth)

plotter = Plotter()
plotter.plot_ground_truths({ground_truth}, [0, 3])   # indices of x,y in state vector
plotter.plot_tracks({track}, [0, 3])
plotter.plot_measurements(
    [det for det_set in detections for det in det_set],
    [0, 3],
    measurement_model=measurement_model
)
plt.text(0.05, 0.95, f"Average OSPA: {avg_ospa:.2f}m",
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.grid()
plotter.fig.show()
plt.show()






















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
