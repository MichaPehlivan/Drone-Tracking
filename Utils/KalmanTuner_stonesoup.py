from datetime import timedelta
import itertools
import random

import numpy as np

from Evaluation_Metrics.Single_Target_Evaluation import mean_ospa_stonesoup
from Kalman_Filters import UnscentedKalmanFilter
from Utils.Wrapper_Functions import SimulatorPolarMultitarget_stonesoup
from Utils.Wrapper_Functions import UKFPredictor, UKFUpdater
from stonesoup.types.state import GaussianState
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.distance import DistanceHypothesiser


def TuneUKF_stonesoup(
    f,
    h,
    x0,
    var_phi,
    var_r,
    beta,
    kappa,
    dt,
    start_time,
    shared_config,
    drones_params,
    N,
):
    Q_base = np.eye(6)

    P0_base = np.eye(6)

    R_base = np.array([[var_phi, 0], [0, var_r]])

    Q_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    P_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    R_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    alpha_space = np.array([1e-3, 1e-2, 0.1, 0.5, 1, 2])
    combinations = list(itertools.product(Q_space, P_space, R_space, alpha_space))
    combinations = random.sample(combinations, 200)

    data = [
        SimulatorPolarMultitarget_stonesoup(
            **shared_config, drone_configs=drones_params
        )
        for _ in range(N)
    ]

    scores = {}

    i = 0
    for Q_mul, P0_mul, R_mul, alpha in combinations:
        i += 1
        print(f"{i}/200")
        Q = Q_base * Q_mul
        P0 = P0_base * P0_mul
        R = R_base * R_mul
        average_score = 0
        try:
            for detections, ground_truths in data:
                ukf = UnscentedKalmanFilter(
                    f=f,
                    h=h,
                    Q=Q,
                    R=R,
                    x0=x0,
                    P0=P0,
                    alpha=alpha,
                    beta=beta,
                    kappa=kappa,
                )

                predictor = UKFPredictor(ukf=ukf)
                updater = UKFUpdater(ukf=ukf)

                prior = GaussianState(
                    state_vector=x0,
                    covar=ukf.P,
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
                    hypotheses = data_associator.associate(
                        tracks, measurements, timestamp
                    )
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
                        measurements - associated_measurements,
                        start_time + timedelta(seconds=dt * n),
                    )
                    all_tracks |= tracks

                mean_ospa = mean_ospa_stonesoup(
                    track=all_tracks, ground_truth=ground_truths
                )
                average_score += mean_ospa
            scores[f"|Q|={Q_mul}, |R|={R_mul}, |P0|={P0_mul}, alpha={alpha}"] = (
                average_score / N
            )
        except np.linalg.LinAlgError:
            scores[f"|Q|={Q_mul}, |R|={R_mul}, |P0|={P0_mul}, alpha={alpha}"] = 0
            continue

    min_params = min((k for k in scores if scores[k] != 0), key=lambda k: scores[k])
    print(f"Minimum OSPA for UKF = {scores[min_params]}, with {min_params}")
