from stonesoup.predictor.base import Predictor
from stonesoup.updater.base import Updater
from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.state import State
from stonesoup.types.array import StateVector
from Track_Simulation import simulateRandomAccelHoverTrackPolar, simulateRandomAccelTrackPolar
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from datetime import datetime, timedelta
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.update import GaussianStateUpdate
import numpy as np


class MyUKFPredictor(Predictor):
    def predict(self, prior, timestamp, **kwargs):
        # 1. Instantiate your UKF using the parameters stored in the model
        # We assume your transition function 'f' is stored in self.transition_model
        ukf = UnscentedKalmanFilter(
            f=self.transition_model.function,
            h=None, Q=self.transition_model.covar(), R=None,
            x0=prior.state_vector, P0=prior.covar,
            alpha=2, beta=2, kappa=0  # Use your tuned params here
        )

        # 2. Call your existing logic
        pred_x, _ = ukf.predict()

        # 3. Return the Stone Soup object
        return GaussianStatePrediction(pred_x, ukf.P, timestamp=timestamp)


def SimulatorPolar_stonesoup(**kwargs):

    start_time = datetime.now()
    sim_function = kwargs.pop('sim_function', simulateRandomAccelTrackPolar)
    dt = kwargs.get('dt')
    measurement_model = kwargs.pop("measurement_model")

    measurements, true_track = sim_function(**kwargs)

    all_detections = []
    ground_truth = GroundTruthPath()

    for i in range(measurements.shape[1]):
        timestamp = start_time + timedelta(seconds=i * dt)

        x_true = true_track[0, i]
        y_true = true_track[1, i]

        # Create Detection
        det = Detection(
            state_vector=StateVector([ measurements[1,i], measurements[0,i] ]), # reverse order as stonesoup expects [phi, r] instead of r,phi
            timestamp=timestamp,
            measurement_model=measurement_model
        )
        all_detections.append({det})

        # Create Ground Truth State and add to Path
        truth_state = State(
            state_vector=StateVector([x_true, 0, 0, y_true, 0, 0]),
            timestamp=timestamp
        )
        ground_truth.append(truth_state)

    return all_detections, ground_truth



# def RandomAccelHoverTrackPolar_stonesoup(**kwargs):
#
#     dt = kwargs.pop('dt', 1.0)
#     measurements, true_track = simulateRandomAccelHoverTrackPolar(**kwargs)
#
#     for i in range(measurements.shape[1]):
#         yield {Detection(StateVector(measurements[:, i]), timestamp=i*dt)}
#^^ yield cool voor testen van de real time versie!!
