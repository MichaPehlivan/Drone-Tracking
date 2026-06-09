from datetime import datetime, time, timedelta
import itertools
import random

import numpy as np
import optuna
from func_timeout import func_timeout, FunctionTimedOut

from Evaluation_Metrics.Single_Target_Evaluation import mean_ospa_stonesoup
from Kalman_Filters import UnscentedKalmanFilter
from Kalman_Filters.Kalman_Filters import ExtendedKalmanFilter, UCMKalmanFilter
from Utils import ReadDetections
from Utils.DecodeGPS import gps_to_ground_truth, interpolate_ground_truth
from Utils.Wrapper_Functions import SimulatorPolarMultitarget_stonesoup
from Utils.Wrapper_Functions import UKFPredictor, UKFUpdater
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.state import GaussianState
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.distance import DistanceHypothesiser

from Utils.Wrapper_Functions.StoneSoup_Wrappers import (
    EKFPredictor,
    EKFUpdater,
    UCMKFPredictor,
    UCMKFUpdater,
)


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

    Q_base = Q_generator(dt)

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
                    P0=P0,
                    alpha=alpha,
                    beta=beta,
                    kappa=kappa,
                    x0=x0,
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


def optimize_UKF_stonesoup(
    f,
    h,
    x0,
    beta,
    kappa,
    var_r,
    var_phi,
    dt,
    start_time,
    shared_config,
    drones_params,
    N,
):
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

    Q_base = Q_generator(dt)

    R_base = np.array([[var_phi, 0], [0, var_r]])

    P0_base = np.eye(6)

    evaluation_data = []
    for _ in range(N):
        detections, groundTruths = SimulatorPolarMultitarget_stonesoup(
            **shared_config, drone_configs=drones_params
        )
        evaluation_data.append((detections, groundTruths))

    def objective(trial):
        # Sample parameters continuously on a log scale
        Q_mul = trial.suggest_float("|Q|", 1e-4, 1e3, log=True)
        R_mul = trial.suggest_float("|R|", 1e-4, 1e3, log=True)
        P0_mul = trial.suggest_float("|P0|", 1e-4, 1e3, log=True)
        alpha = trial.suggest_float("alpha", 1e-4, 1e3, log=True)

        Q = Q_base * Q_mul
        R = R_base * R_mul
        P0 = P0_base * P0_mul

        total_ospa = 0
        for detections, ground_truths in evaluation_data:
            ukf = UnscentedKalmanFilter(
                f=f,
                h=h,
                Q=Q,
                R=R,
                P0=P0,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                x0=x0,
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

            # Iterate over measurements to implement the recursive structure.
            try:
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

                total_ospa += mean_ospa_stonesoup(
                    track=all_tracks, ground_truth=ground_truths
                )
            except np.linalg.LinAlgError:
                # Penalize failed filter runs heavily so the optimizer avoids this region
                return float("inf")

        return total_ospa / len(evaluation_data)

    # Create and run the study
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective, n_trials=150, n_jobs=-1
    )  # 100 intelligent guesses instead of 512 blind ones

    print(f"Minimum OSPA for UKF = {study.best_value}, with {study.best_params}")


def optimize_filter_on_data(
    x0,
    sigma_r,
    sigma_phi,
    dt,
    start_time,
    model,
    filter,
    association_distance,
    deletion_covariance,
    initiation_points,
    detections_path,
    gps_path,
):
    # Initialize the functions and matrices for the Kalman filter.
    F_generator = lambda dt: (
        np.array(
            [
                [1, dt, 0, 0],  # xip
                [0, 1, 0, 0],  # vx
                [0, 0, 1, dt],  # y
                [0, 0, 0, 1],  # vy
            ]
        )
        if model == "cv"
        else np.array(
            [
                [1, dt, 0.5 * dt**2, 0, 0, 0],  # x
                [0, 1, dt, 0, 0, 0],  # vx
                [0, 0, 1, 0, 0, 0],  # ax
                [0, 0, 0, 1, dt, 0.5 * dt**2],  # y
                [0, 0, 0, 0, 1, dt],  # vy
                [0, 0, 0, 0, 0, 1],  # ay
            ]
        )
    )
    F = F_generator(dt) if filter == "ucmkf" else lambda x: F_generator(dt)

    f = lambda x: np.dot(F_generator(dt), x)

    h = (
        np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        if filter == "ucmkf" and model == "cv"
        else (
            np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
            if filter == "ucmkf"
            else lambda x: (
                np.array([np.arctan2(x[2], x[0]), np.sqrt(x[0] ** 2 + x[2] ** 2)])
                if model == "cv"
                else np.array([np.arctan2(x[3], x[0]), np.sqrt(x[0] ** 2 + x[3] ** 2)])
            )
        )
    )

    H = lambda x: (
        np.array(
            [
                [
                    (-x[2]) / (1e-9 + x[0] ** 2 + x[2] ** 2),
                    0,
                    x[0] / (1e-9 + x[0] ** 2 + x[2] ** 2),
                    0,
                ],
                [
                    x[0] / np.sqrt(1e-9 + x[0] ** 2 + x[2] ** 2),
                    0,
                    x[2] / np.sqrt(1e-9 + x[0] ** 2 + x[2] ** 2),
                    0,
                ],
            ]
        )
        if model == "cv"
        else np.array(
            [
                [
                    (-x[3]) / (x[0] ** 2 + x[3] ** 2),
                    0,
                    0,
                    x[0] / (x[0] ** 2 + x[3] ** 2),
                    0,
                    0,
                ],
                [
                    x[0] / np.sqrt(x[0] ** 2 + x[3] ** 2),
                    0,
                    0,
                    x[3] / np.sqrt(x[0] ** 2 + x[3] ** 2),
                    0,
                    0,
                ],
            ]
        )
    )
    # Process noise matrix.
    # IMPORTANT: the process noise is dependent on dt
    var_a = 2
    Q1D_generator = lambda dt: (
        var_a
        * np.array(
            [
                [(dt**3 / 3), (dt**2 / 2)],
                [(dt**2 / 2), dt],
            ]
        )
        if model == "cv"
        else var_a
        * np.array(
            [
                [(dt**5 / 20), (dt**4 / 8), (dt**3 / 6)],
                [(dt**4 / 8), (dt**3 / 3), (dt**2 / 2)],
                [(dt**3 / 6), (dt**2 / 2), dt],
            ]
        )
    )
    Q_generator = lambda dt: (
        np.block(
            [
                [Q1D_generator(dt), np.zeros((2, 2))],
                [np.zeros((2, 2)), Q1D_generator(dt)],
            ]
        )
        if model == "cv"
        else np.block(
            [
                [Q1D_generator(dt), np.zeros((3, 3))],
                [np.zeros((3, 3)), Q1D_generator(dt)],
            ]
        )
    )

    Q_base = Q_generator(dt)

    # Starting error covariance (should be on the higher side to quickly settle in towards the correct values. i.e high uncertainty to start with :))
    P0_base = np.eye(4) if model == "cv" else np.eye(6)

    # Initialize measurement error matrix.
    R_base = np.array([[sigma_phi**2, 0], [0, sigma_r**2]])

    # Define measurement_model
    measurement_model = (
        CartesianToBearingRange(ndim_state=4, mapping=(0, 2), noise_covar=R_base)
        if model == "cv"
        else CartesianToBearingRange(ndim_state=6, mapping=(0, 3), noise_covar=R_base)
    )

    detections = ReadDetections(
        filepath=detections_path,
        measurement_model=measurement_model,
        dt=dt,
        start_time=start_time,
    )
    time_duration_recording_s = timedelta(seconds=dt) * len(detections)
    print(time_duration_recording_s)

    ground_truth = gps_to_ground_truth(
        filepath=gps_path,
        model="cv",
        start_time=start_time,
    )

    def objective(trial):
        # Sample parameters continuously on a log scale
        Q_mul = trial.suggest_float("|Q|", 1e-4, 1e3, log=True)
        R_mul = trial.suggest_float("|R|", 1e-4, 1e3, log=True)
        P0_mul = trial.suggest_float("|P0|", 1e-4, 1e3, log=True)
        alpha = 1  # (
        #     trial.suggest_float("alpha", 1e-4, 1e3, log=True)
        #     if filter == "ukf"
        #     else trial.suggest_float("alpha", 0, 0, log=True)
        # )

        Q = Q_base * Q_mul
        R = R_base * R_mul
        P0 = P0_base * P0_mul

        # ___Initialize the filter____
        kf = (
            UCMKalmanFilter(
                F=F,
                H=h,
                Q=Q,
                P0=P0,
                sigma_r=sigma_r,
                sigma_phi=sigma_phi,
                x0=x0,
            )
            if filter == "ucmkf"
            else (
                ExtendedKalmanFilter(f=f, h=h, F=F, H=H, Q=Q, R=R, P0=P0, x0=x0)
                if filter == "ekf"
                else UnscentedKalmanFilter(
                    f=f,
                    h=h,
                    Q=Q,
                    R=R,
                    P0=P0,
                    alpha=alpha,
                    beta=2,
                    kappa=0,
                    x0=x0,
                )
            )
        )

        predictor = (
            UCMKFPredictor(ucmkf=kf)
            if filter == "ucmkf"
            else EKFPredictor(ekf=kf) if filter == "ekf" else UKFPredictor(ukf=kf)
        )
        updater = (
            UCMKFUpdater(ucmkf=kf)
            if filter == "ucmkf"
            else EKFUpdater(ekf=kf) if filter == "ekf" else UKFUpdater(ukf=kf)
        )

        prior = GaussianState(
            state_vector=x0,
            covar=kf.P,
            timestamp=start_time,
        )

        hypothesiser = DistanceHypothesiser(
            predictor,
            updater,
            measure=Mahalanobis(),
            missed_distance=association_distance,
        )

        data_associator = GlobalNearestNeighbour(hypothesiser)

        deleter = CovarianceBasedDeleter(covar_trace_thresh=deletion_covariance)

        initiator = MultiMeasurementInitiator(
            prior_state=prior,
            deleter=deleter,
            data_associator=data_associator,
            updater=updater,
            min_points=initiation_points,
        )

        def run_tracking():
            tracks, all_tracks = set(), set()
            timesteps = []

            trial_start_time = datetime.now()

            for n, measurements in enumerate(detections):
                if datetime.now() - trial_start_time > timedelta(seconds=5):
                    raise optuna.exceptions.TrialPruned(
                        f"Trial pruned: Exceeded 5s limit."
                    )

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
                    measurements - associated_measurements, timestamp
                )
                all_tracks |= tracks

            interpolated_ground_truth = interpolate_ground_truth(
                ground_truth[0], timesteps, model
            )

            # Evaluate using the ALIGNED ground truth
            total_ospa = mean_ospa_stonesoup(
                all_tracks, interpolated_ground_truth, in_frame=True
            )

            return total_ospa

        try:
            total_ospa = func_timeout(5, run_tracking)
            return total_ospa
        except FunctionTimedOut:
            raise optuna.exceptions.TrialPruned("Trial pruned: Hard timeout reached.")

    # Create and run the study
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective, n_trials=150
    )  # 100 intelligent guesses instead of 512 blind ones

    print(f"Minimum OSPA for {filter} = {study.best_value}, with {study.best_params}")
