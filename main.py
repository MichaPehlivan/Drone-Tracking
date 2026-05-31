# External Imports
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Internal Packages

from Evaluation_Metrics import ospa_stonesoup
from Kalman_Filters import UnscentedKalmanFilter, ExtendedKalmanFilter
from Track_Simulation import (
    simulateLinearTrack,
    simulateLinearTrackPolar,
    simulateRandomAccelTrackPolar,
    simulateRandomAccelHoverTrackPolar,
)
from Utils.Wrapper_Functions import (
    SimulatorPolar_stonesoup,
    SimulatorPolarMultitarget_stonesoup,
    EKFUpdater,
    EKFPredictor,
    UKFPredictor,
    UKFUpdater,
    # CustomDistanceMeasure
)
from Utils import ReadDetections, ReadAndClusterDetections

# Stonesoup imports
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.dataassociator.probability import JPDA
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.plotter import Plotter

# Initialize values
dt = 0.0112 * 25
# dt = 0.1


# Initialize the functions and matrices for the Kalman filter.
# constant velcity (CV) model
# F_generator = lambda dt: np.array(
#     [
#         [1, dt, 0, 0],  # x
#         [0, 1, 0, 0],  # vx
#         [0, 0, 1, dt],  # y
#         [0, 0, 0, 1],  # vy
#     ]
# )
# F = lambda x: F_generator(dt)

# f = lambda x: np.dot(F_generator(dt), x)

# h = lambda x: np.array([np.arctan2(x[2], x[0]), np.sqrt(x[0] ** 2 + x[2] ** 2)])

# H = lambda x: np.array(
#     [
#         [
#             (-x[2]) / (x[0] ** 2 + x[2] ** 2),
#             0,
#             x[0] / (x[0] ** 2 + x[2] ** 2),
#             0,
#         ],
#         [
#             x[0] / np.sqrt(x[0] ** 2 + x[2] ** 2),
#             0,
#             x[2] / np.sqrt(x[0] ** 2 + x[2] ** 2),
#             0,
#         ],
#     ]
# )

# # Process noise matrix.
# # IMPORTANT: the process noise is dependent on dt
# # Q = 0.01 * np.eye(6)
# var_a = 0.5
# Q1D_generator = lambda dt: var_a * np.array(
#     [
#         [(dt**3 / 3), (dt**2 / 2)],
#         [(dt**2 / 2), dt],
#     ]
# )
# Q_generator = lambda dt: np.block(
#     [[Q1D_generator(dt), np.zeros((2, 2))], [np.zeros((2, 2)), Q1D_generator(dt)]]
# )

# Q = Q_generator(dt)

# x0 = np.array([[1], [0], [1], [0]])

# # Starting error covariance (should be on the higher side to quickly settle in towards the correct values. i.e high uncertainty to start with :))
# P0 = 1 * np.eye(4)

# constant accelaration (CA) model
F_generator = lambda dt: np.array(
    [
        [1, dt, 0.5 * dt**2, 0, 0, 0],  # x
        [0, 1, dt, 0, 0, 0],  # vx
        [0, 0, 1, 0, 0, 0],  # ax
        [0, 0, 0, 1, dt, 0.5 * dt**2],  # y
        [0, 0, 0, 0, 1, dt],  # vy
        [0, 0, 0, 0, 0, 1],  # ay
    ]
)
F = lambda x: F_generator(dt)

f = lambda x: np.dot(F_generator(dt), x)

h = lambda x: np.array([np.arctan2(x[3], x[0]), np.sqrt(x[0] ** 2 + x[3] ** 2)])

H = lambda x: np.array(
    [
        [(-x[3]) / (x[0] ** 2 + x[3] ** 2), 0, 0, x[0] / (x[0] ** 2 + x[3] ** 2), 0, 0],
        [
            x[0] / np.sqrt(x[0] ** 2 + x[3] ** 2),
            0,
            0,
            x[3] / np.sqrt(x[0] ** 2 + x[3] ** 2),
            0,
            0,
        ],
    ]
)

var_a = 0.5
Q1D_generator = lambda dt: var_a * np.array(
    [
        [(dt**5 / 20), (dt**4 / 8), (dt**3 / 6)],
        [(dt**4 / 8), (dt**3 / 3), (dt**2 / 2)],
        [(dt**3 / 6), (dt**2 / 2), dt],
    ]
)
Q_generator = lambda dt: np.block(
    [[Q1D_generator(dt), np.zeros((3, 3))], [np.zeros((3, 3)), Q1D_generator(dt)]]
)

Q = Q_generator(dt)

x0 = np.array([[1], [0], [0], [1], [0], [0]])

# Starting error covariance (should be on the higher side to quickly settle in towards the correct values. i.e high uncertainty to start with :))
P0 = 5 * np.eye(6)

# Define variances in measurement dimensions.
range_sigma = 3
azimuth_sigma = np.deg2rad(3)

# convert to variance.
var_r = range_sigma**2
var_phi = azimuth_sigma**2

# Initialize measurement error matrix.
R = 1 * np.array([[var_phi, 0], [0, var_r]])

# Define measurement_model
# measurement_model = CartesianToBearingRange(ndim_state=4, mapping=(0, 2), noise_covar=R)
measurement_model = CartesianToBearingRange(ndim_state=6, mapping=(0, 3), noise_covar=R)

# start the clock for easy timestamp management
start_time = datetime.now()

# Get the detections from Abdullahs group
detections = ReadDetections(
    filepath="Data/flight2/flight2-sidetoside-tiny_tinyrad_master_1_rd_guided_mvdr_range_angle_velocity_detections_ndoppler250.csv",
    measurement_model=measurement_model,
    dt=dt,
    start_time=start_time,
)


# ___Initialize the filter____
# ekf = ExtendedKalmanFilter(f=f, h=h, F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
ukf = UnscentedKalmanFilter(f=f, h=h, Q=Q, R=R, x0=x0, P0=P0, alpha=1, beta=2, kappa=0)

# predictor = EKFPredictor(ekf=ekf)
# updater = EKFUpdater(ekf=ekf)
predictor = UKFPredictor(ukf=ukf)
updater = UKFUpdater(ukf=ukf)

# prior = GaussianState(
#     state_vector=x0,
#     covar=ekf.P,
#     timestamp=start_time,
# )

prior = GaussianState(
    state_vector=x0,
    covar=ukf.P,
    timestamp=start_time,
)


hypothesiser = DistanceHypothesiser(
    predictor, updater, measure=Mahalanobis(), missed_distance=2
)

data_associator = GlobalNearestNeighbour(hypothesiser)

deleter = CovarianceBasedDeleter(covar_trace_thresh=25.0)

initiator = MultiMeasurementInitiator(
    prior_state=prior,
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
    min_points=5,
)
# ________________________


tracks, all_tracks = set(), set()
timesteps = []
previous_timestamp = None

for n, measurements in enumerate(detections):

    timestamp = start_time + timedelta(seconds=dt * n)
    timesteps.append(timestamp)

    hypotheses = data_associator.associate(tracks, measurements, timestamp)
    associated_measurements = set()

    for track in tracks:
        hypothesis = hypotheses[track]

        if hypothesis.measurement:
            post = updater.update(hypothesis)
            track.append(post)

            associated_measurements.add(hypothesis.measurement)

        else:
            track.append(hypothesis.prediction)

    tracks -= deleter.delete_tracks(tracks)
    tracks |= initiator.initiate(measurements - associated_measurements, timestamp)
    all_tracks |= tracks


# Plotting
plotter = Plotter()
# plotter.plot_ground_truths(ground_truths, [0, 3])   # indices of x,y in state vector
plotter.plot_tracks(all_tracks, [0, 3])
plotter.plot_measurements(
    [det for det_set in detections for det in det_set],
    [0, 3],
    measurement_model=measurement_model,
)

ax = plt.gca()
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
data_min = min(x_min, y_min)
data_max = max(x_max, y_max)
# ax.set_xlim(data_min, data_max)
# ax.set_ylim(data_min, data_max)
ax.set_xlim(-10, 50)
ax.set_ylim(-30, 30)
ax.set_aspect("equal", adjustable="box")

plt.grid()
plotter.fig.show()
plt.show()
print(len(timesteps))
from stonesoup.plotter import AnimatedPlotterly

print("test")
plotter = AnimatedPlotterly(timesteps, tail_length=1)
print("test")
plotter.fig.update_layout(width=700, height=700)
plotter.plot_tracks(all_tracks, [0, 3], uncertainty=True)
# plotter.plot_ground_truths(ground_truths, [0, 3])
plotter.plot_measurements(detections, [0, 3])
print("test")
plotter.fig.show()
