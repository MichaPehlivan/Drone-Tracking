from stonesoup.predictor.base import Predictor
from stonesoup.updater.base import Updater
from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.array import StateVector
from Track_Simulation import simulateRandomAccelHoverTrackPolar, simulateRandomAccelTrackPolar
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
import numpy as np
# class MyUKFPredictor(Predictor):
#     def predict(self, prior, timestamp, **kwargs):
#         # 1. Access your existing UKF prediction logic
#         # 'prior' is the state at the previous time step
#         time_interval = timestamp - prior.timestamp
#
#         # 2. RUN YOUR OWN CODE HERE
#         # e.g., new_mean, new_covar = my_custom_ukf_predict(prior.state_vector, prior.covar, time_interval)
#
#         # 3. Return it as a Stone Soup 'GaussianStatePrediction'
#         return GaussianStatePrediction(new_mean, new_covar, timestamp)
#
# class MyUKFUpdater(Updater):
#     def update(self, hypothesis, **kwargs):
#         # 'hypothesis' contains the prediction and the measurement (detection)
#         prediction = hypothesis.prediction
#         detection = hypothesis.measurement
#         # 1. RUN YOUR OWN UPDATE CODE HERE
#         # e.g., post_mean, post_covar = my_custom_ukf_update(prediction, detection)
#         # 2. Return as a Stone Soup 'GaussianStateUpdate'
#         return GaussianStateUpdate(post_mean, post_covar, hypothesis, timestamp=detection.timestamp)

#
# def RandomAccelHoverTrackPolar_stonesoup(**kwargs):
#
#     dt = kwargs.pop('dt', 1.0)
#     measurements, true_track = simulateRandomAccelHoverTrackPolar(**kwargs)
#
#     for i in range(measurements.shape[1]):
#         yield {Detection(StateVector(measurements[:, i]), timestamp=i*dt)}
#^^ yield cool voor testen van de real time versie!!


def RandomAccelHoverTrackPolar_stonesoup(**kwargs):
    dt = kwargs.get('dt')
    measurement_model = kwargs.pop("measurement_model")

    measurements, true_track = simulateRandomAccelHoverTrackPolar(**kwargs)

    all_detections = []

    for i in range(measurements.shape[1]):
        det = Detection(
            state_vector=StateVector(measurements[:, i]),
            timestamp=i * dt,
            measurement_model=measurement_model
        )

        all_detections.append({det})

    return all_detections


def RandomAccelTrackPolar_stonesoup(**kwargs):
    dt = kwargs.get('dt')
    measurement_model = kwargs.pop("measurement_model")

    measurements, true_track = simulateRandomAccelTrackPolar(**kwargs)

    all_detections = []

    for i in range(measurements.shape[1]):
        det = Detection(
            state_vector=StateVector(measurements[:, i]),
            timestamp=i * dt,
            measurement_model=measurement_model
        )

        all_detections.append({det})

    return all_detections