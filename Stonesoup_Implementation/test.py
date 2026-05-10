
import numpy as np
from datetime import datetime, timedelta
start_time = datetime.now().replace(microsecond=0)

# np.random.seed(1991)

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.plotter import AnimatedPlotterly

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

from stonesoup.types.detection import Detection

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(1),
                                                          ConstantVelocity(1)])
timesteps = [start_time]
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
num_datapoints = 21

for k in range(1, num_datapoints):
    timesteps.append(start_time+timedelta(seconds=k))
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))

plotter = AnimatedPlotterly(timesteps, tail_length=1)
plotter.plot_ground_truths(truth, [0, 2])
# plotter.fig.show()


sensor_x = 0
sensor_y = 5

measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(5) ** 2, 0.5]),  # Covariance matrix. 5 degree error in
    # bearing and 1 metre in range
    translation_offset=np.array([[sensor_x], [sensor_y]])  # Offset measurements to location of
    # sensor in cartesian.
)


# We create a set of detections using this sensor model.
measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement, timestamp=state.timestamp,
                                  measurement_model=measurement_model))


# Plot the measurements. Where the model is nonlinear the plotting function uses the inverse
# function to get coordinates.
plotter.plot_measurements(measurements, [0, 2])
# plotter.fig.show()


from stonesoup.predictor.kalman import ExtendedKalmanPredictor
predictor = ExtendedKalmanPredictor(transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model)


# First, we'll create a prior state.
from stonesoup.types.state import GaussianState

prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)


# Next iterate over hypotheses and place in a track.
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]


# Plot the resulting track complete with error ellipses at each estimate.
from matplotlib import pyplot as plt
plotter.plot_tracks(track, [0, 2], uncertainty=True)
plotter.fig.show()


