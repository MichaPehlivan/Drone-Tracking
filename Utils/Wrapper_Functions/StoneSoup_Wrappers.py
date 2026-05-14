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
from stonesoup.base import Property
import numpy as np
from stonesoup.models.transition import TransitionModel


class UKFPredictor(Predictor):
    #inherits the properties of the Predictor class (which I want to emulate)
    ukf: object = Property()
    #Automatically creates an __init__ with magic syntax and stonesoup magic :0

    transition_model: TransitionModel = Property(default=None)

    #deze ook verplicht for some reason, gebruiken hem verder niet

    def predict(self, prior, timestamp=None, **kwargs):

        self.ukf.x = np.array(prior.mean).flatten()
        self.ukf.P = np.array(prior.covar)

        x_pred, sigma_points = self.ukf.predict()

        return GaussianStatePrediction(
            state_vector=x_pred.reshape(-1, 1),
            covar=self.ukf.P,
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

        # Recompute sigma points from current predicted state
        sigma_points = self.ukf.generate_sigma_points(self.ukf.x, self.ukf.P)

        x_updated = self.ukf.update(sigma_points, z)

        return GaussianStateUpdate(
            state_vector=x_updated.reshape(-1, 1),
            covar=self.ukf.P,
            hypothesis=hypothesis,
            timestamp=measurement.timestamp,
        )

    def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):

        self.ukf.x = np.array(predicted_state.mean).flatten()
        self.ukf.P = np.array(predicted_state.covar)

        sigma_points = self.ukf.generate_sigma_points(self.ukf.x, self.ukf.P)
        y = np.array([self.ukf.h(s).flatten() for s in sigma_points])
        y_mean = self.ukf.calculate_polar_mean(y)

        return y_mean.reshape(-1, 1)



def SimulatorPolar_stonesoup(**kwargs):

    start_time = kwargs.pop("start_time")
    sim_function = kwargs.pop('sim_function', simulateRandomAccelTrackPolar)
    measurement_model = kwargs.pop("measurement_model")

    dt = kwargs.get('dt')

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
