from datetime import datetime, timedelta
import numpy as np

from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.transition.linear import LinearGaussianTransitionModel

from stonesoup.types.detection import Detection

#generate transition model
dt = 0.1
x_initial = 0
y_initial = 0

F = np.array([[1, 0, dt, 0],
              [0, 1,  0, dt],
              [0, 0,  1, 0],
              [0, 0, 0, 1]])

Q = 0.0001 * np.eye(4)

x0 = np.array([[x_initial], [y_initial], [1], [1]])

P0 = 1000 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

transition_model = LinearGaussianTransitionModel(
    transition_matrix=F,
    covariance_matrix=Q
)

measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 1),
    noise_covar=np.diag([np.radians(5) ** 2, 1]),  # Covariance matrix. 5 degree error in bearing and 1 metre in range
)



