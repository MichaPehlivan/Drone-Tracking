# External imports
import numpy as np
from datetime import datetime, timedelta

# Predictor and updater for Kalman filter
from stonesoup.predictor.base import Predictor
from stonesoup.updater.base import Updater

# Stonesoup types
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.state import State
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.update import GaussianStateUpdate

from stonesoup.base import Property
from stonesoup.models.transition import TransitionModel
from stonesoup.types.prediction import GaussianMeasurementPrediction


class UKFPredictor(Predictor):
    # inherits the properties of the Predictor class (which I want to emulate)
    ukf: object = Property()
    # Automatically creates an __init__ with magic syntax and stonesoup magic :0

    transition_model: TransitionModel = Property(default=None)
    # deze ook verplicht for some reason, gebruiken hem verder niet

    def predict(self, prior, timestamp=None, **kwargs):

        self.ukf.x = np.array(prior.mean).flatten()
        self.ukf.P = np.array(prior.covar)

        x_pred = self.ukf.predict()

        return GaussianStatePrediction(
            state_vector=StateVector(x_pred.reshape(-1, 1)),
            covar=CovarianceMatrix(self.ukf.P),
            timestamp=timestamp,
        )


class UKFUpdater(Updater):

    ukf: object = Property()

    measurement_model: object = Property(default=None)

    def update(self, hypothesis, **kwargs):

        prediction = hypothesis.prediction
        measurement = hypothesis.measurement

        # Sync UKF with the predicted state
        self.ukf.x = np.array(prediction.mean).flatten()
        self.ukf.P = np.array(prediction.covar)

        # Extract measurement vector
        z = np.array(measurement.state_vector).flatten()

        x_updated, _, _ = self.ukf.update(z)

        return GaussianStateUpdate(
            state_vector=StateVector(x_updated.reshape(-1, 1)),
            covar=CovarianceMatrix(self.ukf.P),
            hypothesis=hypothesis,
            timestamp=measurement.timestamp,
        )

    def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):

        self.ukf.x = np.array(predicted_state.mean).flatten()
        self.ukf.P = np.array(predicted_state.covar)

        sigma_points = self.ukf.generate_sigma_points(self.ukf.x, self.ukf.P)
        y = np.array([self.ukf.h(s).flatten() for s in sigma_points])
        y_mean = self.ukf.calculate_polar_mean(y)

        weights = self.ukf.wm
        y_diff = y - y_mean
        S = np.zeros((y_mean.shape[0], y_mean.shape[0]))

        for idx in range(len(sigma_points)):
            S += weights[idx] * np.outer(y_diff[idx], y_diff[idx])
        S += self.ukf.R

        # Covariance matrix necessary for the Mahalanobis distance measure, wanted to try it
        # TODO: justify why this is the right covariance matrix, just same as error matrix x-xhat except scaled with weights.
        return GaussianMeasurementPrediction(
            state_vector=StateVector(y_mean.reshape(-1, 1)),
            covar=CovarianceMatrix(S),
            timestamp=predicted_state.timestamp,
        )


""" 
class CustomDistanceMeasure:
    def __call__(self, state1, state2):

        vec1 = getattr(state1, 'state_vector', state1)
        vec2 = getattr(state2, 'state_vector', state2)

        v1 = np.asarray(vec1).flatten()
        v2 = np.asarray(vec2).flatten()
        return np.linalg.norm(v1 - v2)
"""


def SimulatorPolar_stonesoup(**kwargs):

    start_time = kwargs.pop("start_time")
    sim_function = kwargs.pop("sim_function")
    measurement_model = kwargs.pop("measurement_model")

    dt = kwargs.get("dt")

    measurements, true_track = sim_function(**kwargs)

    all_detections = []
    ground_truth = GroundTruthPath()

    for i in range(measurements.shape[1]):
        timestamp = start_time + timedelta(seconds=i * dt)

        x_true = true_track[0, i]
        y_true = true_track[1, i]

        # Create Detection
        det = Detection(
            state_vector=StateVector(
                [measurements[1, i], measurements[0, i]]
            ),  # reverse order as stonesoup expects [phi, r] instead of r,phi
            timestamp=timestamp,
            measurement_model=measurement_model,
        )
        all_detections.append({det})

        # Create Ground Truth State and add to Path
        truth_state = State(
            state_vector=StateVector([x_true, 0, 0, y_true, 0, 0]), timestamp=timestamp
        )
        ground_truth.append(truth_state)

    return all_detections, ground_truth


def SimulatorPolarMultitarget_stonesoup(**kwargs):
    add_clutter = kwargs.pop("add_clutter", False)
    start_time = kwargs.pop("start_time")
    sim_function = kwargs.pop("sim_function")
    measurement_model = kwargs.pop("measurement_model")
    dt = kwargs.get("dt")

    drone_configs = kwargs.pop(
        "drone_configs", [{}, {}]
    )  # Defaults to two empty drones
    measurements_list = []
    true_tracks_list = []
    delays_list = []

    for config in drone_configs:

        delay_steps = config.pop("delay_steps", 0)
        delays_list.append(delay_steps)

        combined_config = kwargs | config

        meas, track = sim_function(**combined_config)

        measurements_list.append(meas)
        true_tracks_list.append(track)

    num_drones = len(drone_configs)
    num_time_steps = measurements_list[0].shape[1]

    all_detections = []
    ground_truths = [GroundTruthPath() for _ in range(num_drones)]

    for i in range(num_time_steps):
        timestamp = start_time + timedelta(seconds=i * dt)
        time_step_detections = set()

        for drone in range(num_drones):
            meas = measurements_list[drone]
            track = true_tracks_list[drone]
            delay = delays_list[drone]

            local_i = i - delay

            if 0 <= local_i < meas.shape[1]:

                det = Detection(
                    state_vector=StateVector([meas[1, local_i], meas[0, local_i]]),
                    timestamp=timestamp,
                    measurement_model=measurement_model,
                )
                time_step_detections.add(det)

                if add_clutter:
                    clutter_chance = 0.2
                    if np.random.uniform(0, 1) < clutter_chance:

                        rand_x = np.random.uniform(-30, 150)
                        rand_y = np.random.uniform(-30, 150)
                        rand_phi = np.arctan2(rand_y, rand_x)
                        rand_R = np.sqrt(rand_x**2 + rand_y**2)
                        det = Detection(
                            state_vector=StateVector([rand_phi, rand_R]),
                            timestamp=timestamp,
                            measurement_model=measurement_model,
                        )
                        time_step_detections.add(det)

                x_true = track[0, local_i]
                y_true = track[1, local_i]

                truth_state = State(
                    state_vector=StateVector([x_true, 0, 0, y_true, 0, 0]),
                    timestamp=timestamp,
                )
                ground_truths[drone].append(truth_state)

        all_detections.append(time_step_detections)

    return all_detections, ground_truths


# def RandomAccelHoverTrackPolar_stonesoup(**kwargs):
#
#     dt = kwargs.pop('dt', 1.0)
#     measurements, true_track = simulateRandomAccelHoverTrackPolar(**kwargs)
#
#     for i in range(measurements.shape[1]):
#         yield {Detection(StateVector(measurements[:, i]), timestamp=i*dt)}
# ^^ yield cool voor testen van de real time versie!!
