# External Imports
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Internal Packages
from Evaluation_Metrics import ospa_stonesoup
from Kalman_Filters import UnscentedKalmanFilter, ExtendedKalmanFilter
from Utils.KalmanTuner import TuneEKF, TuneUKF
from Track_Simulation import (
    simulateLinearTrack,
    simulateLinearTrackPolar,
    simulateRandomAccelTrackPolar,
    simulateRandomAccelHoverTrackPolar,
)
from Utils.KalmanTuner_stonesoup import TuneUKF_stonesoup, optimize_UKF_stonesoup
from Utils.KalmanBenchmarker import BenchmarkUKF_stonesoup
from Utils.Wrapper_Functions import (
    SimulatorPolar_stonesoup,
    SimulatorPolarMultitarget_stonesoup,
    EKFUpdater,
    EKFPredictor,
)

# Stonesoup imports
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.state import GaussianState
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.plotter import Plotter

from Utils.Wrapper_Functions.StoneSoup_Wrappers import (
    SimulatorPolarMultitarget_stonesoup,
)

# Initialize values
dt = 0.4
x_initial = 5
y_initial = 5

num_datapoints = 100

# Initialize the functions and matrices for the Kalman filter.

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


h_cartesian = lambda x: np.array([x[0], x[3]])

H_cartesian = lambda x: np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

h_polar = lambda x: np.array([np.arctan2(x[3], x[0]), np.sqrt(x[0] ** 2 + x[3] ** 2)])

H_polar = lambda x: np.array([
    [
        (-x[3]) / (x[0]**2 + x[3]**2),
        0,
        0,
        x[0]  / (x[0]**2 + x[3]**2),
        0,
        0
    ],
    [
        x[0] / np.sqrt(x[0]**2 + x[3]**2),
        0,
        0,
        x[3] / np.sqrt(x[0]**2 + x[3]**2),
        0,
        0
    ],
])

Q1D_generator = lambda dt: np.array(
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


x0 = np.array([[x_initial], [1], [0], [y_initial], [1], [0]])
# Constant acceleration (CA)
P0 = np.eye(6)

range_sigma = 1
azimuth_sigma = np.deg2rad(3)
var_r = range_sigma**2
var_phi = azimuth_sigma**2

R = np.array([[var_phi, 0], [0, var_r]])

# Define measurement_model
measurement_model = CartesianToBearingRange(ndim_state=6, mapping=(0, 3), noise_covar=R)

start_time = datetime.now()

shared_config = {
    "sim_function": simulateRandomAccelTrackPolar,
    "start_time": start_time,
    "measurement_model": measurement_model,
    "num_datapoints": num_datapoints,
    "dt": dt,
    "sigma_r": range_sigma,
    "sigma_phi": azimuth_sigma,
    "add_clutter": True,
}

drones_params = [
    {
        "x0": x_initial,
        "y0": y_initial,
        "v_x": 5.0,
        "v_y": 5.0
    },
    {
        "x0": x_initial + 100,
        "y0": y_initial,
        "v_x": -5,
        "v_y": 5
    },
    {
        "x0": x_initial + 100,
        "y0": y_initial + 70,
        "v_x": -5,
        "v_y": 1,
        "delay_steps": 10,
    },
    {
        "x0": x_initial + 150,
        "y0": y_initial + 70,
        "v_x": 1,
        "v_y": 4,
        "delay_steps": 50,
    },
    # {
    #     "x0": x_initial + 30,
    #     "y0": y_initial + 90,
    #     "v_x": 0,
    #     "v_y": -4,
    #     "delay_steps": 30,
    # },
]

detections, ground_truths = SimulatorPolarMultitarget_stonesoup(
    **shared_config, drone_configs=drones_params
)


# ukf = UnscentedKalmanFilter(
#     f=f, h=h_polar, Q=Q_UKF, R=R_UKF, x0=x0, P0=P0_UKF, alpha=alpha, beta=2, kappa=0
# )
ekf = ExtendedKalmanFilter(f=f, h=h_polar, F=F, H=H_polar, Q=Q, R=R, x0=x0, P0=P0)

predictor = EKFPredictor(ekf=ekf)
updater = EKFUpdater(ekf=ekf)

prior = GaussianState(
    state_vector=x0,
    covar=ekf.P,
    timestamp=start_time,
)

hypothesiser = DistanceHypothesiser(
    predictor, updater, measure=Mahalanobis(), missed_distance=5
)

data_associator = GlobalNearestNeighbour(hypothesiser)

deleter = UpdateTimeStepsDeleter(time_steps_since_update=3)

initiator = MultiMeasurementInitiator(
    prior_state=prior,  # dummy
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
    min_points=4,
)


tracks, all_tracks = set(), set()
timesteps = []

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
    tracks |= initiator.initiate(
        measurements - associated_measurements, start_time + timedelta(seconds=dt * n)
    )
    all_tracks |= tracks


ospa_stonesoup(track=all_tracks, ground_truth=ground_truths)

# Plotting
plotter = Plotter()
plotter.plot_ground_truths(ground_truths, [0, 3])  # indices of x,y in state vector
plotter.plot_tracks(all_tracks, [0, 3])
plotter.plot_measurements(
    [det for det_set in detections for det in det_set],
    [0, 3],
    measurement_model=measurement_model,
)
# plt.text(0.05, 0.95, f"Average OSPA: {avg_ospa:.2f}m",
#          transform=plt.gca().transAxes,
#          fontsize=12,
#          verticalalignment='top',
#          bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax = plt.gca()
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
data_min = min(x_min, y_min)
data_max = max(x_max, y_max)
ax.set_xlim(data_min, data_max)
ax.set_ylim(data_min, data_max)
ax.set_aspect("equal", adjustable="box")

plt.grid()
plotter.fig.show()
plt.show()

from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timesteps, tail_length=1)
plotter.fig.update_layout(width=800, height=800)
plotter.plot_tracks(all_tracks, [0, 3], uncertainty=True)
plotter.plot_ground_truths(ground_truths, [0, 3])
plotter.plot_measurements(detections, [0, 3])
plotter.fig.show()
