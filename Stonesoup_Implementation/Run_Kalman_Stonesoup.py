from datetime import datetime, timedelta
import numpy as np

from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater


def RunStoneSoupKalman(transition_model, measurement_model, x0, P0, measurements, dt):

    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    start_time = datetime.now()
    current_state = GaussianState(x0, P0, timestamp=start_time)

    track = Track()

    for i in range(measurements.shape[1]):
        time_step = start_time + timedelta(seconds=i * dt)

        # Stone Soup Detection object
        detection = Detection(measurements[:, i].reshape(2, 1), timestamp=time_step)

        prediction = predictor.predict(current_state, timestamp=time_step)
        current_state = updater.update(prediction, detection)

        track.append(current_state)

    # Convert track back to your x_history format for your plotting function
    x_history = np.array([state.state_vector for state in track]).T.reshape(4, -1)

    return x_history
