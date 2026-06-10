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


class UCMKFPredictor(Predictor):
    # inherits the properties of the Predictor class (which I want to emulate)
    ucmkf: object = Property()
    # Automatically creates an __init__ with magic syntax and stonesoup magic :0

    transition_model: TransitionModel = Property(default=None)
    # deze ook verplicht for some reason, gebruiken hem verder niet

    def predict(self, prior, timestamp=None, **kwargs):

        self.ucmkf.x = np.array(prior.mean).reshape((-1, 1))
        self.ucmkf.P = np.array(prior.covar)

        x_pred = self.ucmkf.predict()

        return GaussianStatePrediction(
            state_vector=StateVector(x_pred.reshape(-1, 1)),
            covar=CovarianceMatrix(self.ucmkf.P),
            timestamp=timestamp,
        )


class UCMKFUpdater(Updater):

    ucmkf: object = Property()

    measurement_model: object = Property(default=None)

    def update(self, hypothesis, **kwargs):

        prediction = hypothesis.prediction
        measurement = hypothesis.measurement

        self.ucmkf.x = np.array(prediction.mean).reshape((-1, 1))
        self.ucmkf.P = np.array(prediction.covar)

        z = np.array(measurement.state_vector).flatten()

        x_updated = self.ucmkf.update(z)

        return GaussianStateUpdate(
            state_vector=StateVector(x_updated.reshape(-1, 1)),
            covar=CovarianceMatrix(self.ucmkf.P),
            hypothesis=hypothesis,
            timestamp=measurement.timestamp,
        )

    def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):
        x_local = np.array(predicted_state.mean).flatten()
        P_local = np.array(predicted_state.covar)

        z_predicted = np.dot(self.ucmkf.H, x_local)
        z_x = z_predicted[0]
        z_y = z_predicted[1]
        z_predicted_polar = np.array([np.arctan2(z_y, z_x), np.sqrt(z_x**2 + z_y**2)])

        H_polar = (
            np.array(
                [
                    [
                        (-z_y) / (1e-10 + z_x**2 + z_y**2),
                        0,
                        z_x / (1e-10 + z_x**2 + z_y**2),
                        0,
                    ],
                    [
                        z_x / np.sqrt(1e-10 + z_x**2 + z_y**2),
                        0,
                        z_y / np.sqrt(1e-10 + z_x**2 + z_y**2),
                        0,
                    ],
                ]
            )
            if len(x_local) == 4
            else np.array(
                [
                    [
                        (-z_y) / (1e-10 + z_x**2 + z_y**2),
                        0,
                        0,
                        z_x / (1e-10 + z_x**2 + z_y**2),
                        0,
                        0,
                    ],
                    [
                        z_x / np.sqrt(1e-10 + z_x**2 + z_y**2),
                        0,
                        0,
                        z_y / np.sqrt(1e-10 + z_x**2 + z_y**2),
                        0,
                        0,
                    ],
                ]
            )
        )

        # Standard polar measurement noise
        R_polar = np.array([[self.ucmkf.sigma_phi**2, 0], [0, self.ucmkf.sigma_r**2]])

        S = np.dot(H_polar, np.dot(P_local, H_polar.T)) + R_polar

        # Covariance matrix necessary for the Mahalanobis distance measure, wanted to try it
        return GaussianMeasurementPrediction(
            state_vector=StateVector(z_predicted_polar.reshape(-1, 1)),
            covar=CovarianceMatrix(S),
            timestamp=predicted_state.timestamp,
        )


class EKFPredictor(Predictor):
    # inherits the properties of the Predictor class (which I want to emulate)
    ekf: object = Property()
    # Automatically creates an __init__ with magic syntax and stonesoup magic :0

    transition_model: TransitionModel = Property(default=None)
    # deze ook verplicht for some reason, gebruiken hem verder niet

    def predict(self, prior, timestamp=None, **kwargs):

        self.ekf.x = np.array(prior.mean).flatten()
        self.ekf.P = np.array(prior.covar)

        x_pred = self.ekf.predict()

        return GaussianStatePrediction(
            state_vector=StateVector(x_pred.reshape(-1, 1)),
            covar=CovarianceMatrix(self.ekf.P),
            timestamp=timestamp,
        )


class EKFUpdater(Updater):

    ekf: object = Property()

    measurement_model: object = Property(default=None)

    def update(self, hypothesis, **kwargs):

        prediction = hypothesis.prediction
        measurement = hypothesis.measurement

        self.ekf.x = np.array(prediction.mean).flatten()
        self.ekf.P = np.array(prediction.covar)

        z = np.array(measurement.state_vector).flatten()

        x_updated = self.ekf.update(z)

        return GaussianStateUpdate(
            state_vector=StateVector(x_updated.reshape(-1, 1)),
            covar=CovarianceMatrix(self.ekf.P),
            hypothesis=hypothesis,
            timestamp=measurement.timestamp,
        )

    def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):

        x_local = np.array(predicted_state.mean).flatten()
        P_local = np.array(predicted_state.covar)

        z_predicted = self.ekf.h(x_local).flatten()

        H_matrix = self.ekf.H(x_local)

        S = np.dot(H_matrix, np.dot(P_local, H_matrix.T)) + self.ekf.R

        # Covariance matrix necessary for the Mahalanobis distance measure, wanted to try it
        return GaussianMeasurementPrediction(
            state_vector=StateVector(z_predicted.reshape(-1, 1)),
            covar=CovarianceMatrix(S),
            timestamp=predicted_state.timestamp,
        )


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

        x_updated = self.ukf.update(z)

        return GaussianStateUpdate(
            state_vector=StateVector(x_updated.reshape(-1, 1)),
            covar=CovarianceMatrix(self.ukf.P),
            hypothesis=hypothesis,
            timestamp=measurement.timestamp,
        )

    def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):

        x_local = np.array(predicted_state.mean).flatten()
        P_local = np.array(predicted_state.covar)

        sigma_points = self.ukf.generate_sigma_points(x_local, P_local)
        y = np.array([self.ukf.h(s).flatten() for s in sigma_points])
        y_mean = self.ukf.calculate_polar_mean(y)

        y_diff = y - y_mean
        y_diff[:, 0] = self.ukf.normalize_angle(y_diff[:, 0])

        S = np.dot((self.ukf.wc * y_diff.T), y_diff) + self.ukf.R

        # Covariance matrix necessary for the Mahalanobis distance measure, wanted to try it
        return GaussianMeasurementPrediction(
            state_vector=StateVector(y_mean.reshape(-1, 1)),
            covar=CovarianceMatrix(S),
            timestamp=predicted_state.timestamp,
        )


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
            state_vector=StateVector([measurements[0, i], measurements[1, i]]),
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


"""
input:
    add_clutter: standard:False whether to add clutter detections
    start_time: time to provide timestamps
    sim_function = the function used to simulate (SimulateRandomAccelHoverTrackPolar, SimulateLinearTrackPolar etc.)
    measurement_model = stonesoup object containing the variances and the information that it is [bearing, range]
    dt: timestep
    drone_configs: all the parameters for the sim_functions (for multiple drones)
    delay_steps: timedelay for the drone (inside drone config)
output:
    Stone Soup GroundTruth objects
    Stone Soup Detection objects
    
    
    example call!!!:
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
        {"x0": x_initial, "y0": y_initial, "v_x": 5.0, "v_y": 5.0},
        {"x0": x_initial + 100, "y0": y_initial, "v_x": -5, "v_y": 5},
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

"""


def SimulatorPolarMultitarget_stonesoup(**kwargs):
    add_clutter = kwargs.pop("add_clutter", False)
    start_time = kwargs.pop("start_time")
    sim_function = kwargs.pop("sim_function")
    measurement_model = kwargs.pop("measurement_model")
    model = kwargs.pop("model")
    dt = kwargs.get("dt")

    drone_configs = kwargs.pop("drone_configs", [{}, {}])
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
                    state_vector=StateVector([meas[0, local_i], meas[1, local_i]]),
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
                    state_vector=(
                        StateVector([x_true, 0, y_true, 0])
                        if model == "cv"
                        else StateVector([x_true, 0, 0, y_true, 0, 0])
                    ),
                    timestamp=timestamp,
                )
                ground_truths[drone].append(truth_state)

        all_detections.append(time_step_detections)

    return all_detections, ground_truths
