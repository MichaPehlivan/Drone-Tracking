from datetime import timedelta

import numpy as np

from Evaluation_Metrics.Single_Target_Evaluation import get_average_ospa
from Kalman_Filters import UCMKalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from Track_Simulation import simulateRandomAccelTrackPolar, simulateLinearTrackPolar
from Utils.Wrapper_Functions.StoneSoup_Wrappers import (
    SimulatorPolarMultitarget_stonesoup,
    UKFPredictor,
    UKFUpdater,
)
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


def BenchmarkEKF(f, h, F, H, Q, R, x0, P0, var_r, var_phi, dt, N):
    print("Benchmarking EKF on linear track")
    average_score = 0
    for _ in range(N):
        measurements, trueTrack = simulateLinearTrackPolar(
            v_x=1,
            v_y=1,
            x0=5,
            y0=5,
            num_datapoints=60,
            dt=dt,
            sigma_r=np.sqrt(var_r),
            sigma_phi=np.sqrt(var_phi),
        )
        # Define the kalman filter.
        KF = ExtendedKalmanFilter(f, h, F, H, Q, R, x0, P0)

        # Initialize the history array.
        x_history = np.zeros((6, len(measurements[0, :])))

        # Iterate over measurements to implement the recursive structure.
        for i in range(len(measurements[0, :])):
            KF.predict()
            KF.update(measurements[:, i].reshape(2, 1))

            x_history[:, i] = KF.x.reshape(
                6,
            )

        average_ospa = get_average_ospa(x_history, trueTrack)
        average_score += average_ospa
    print(f"Average OSPA for EKF = {average_score / N} over {N} linear tracks")

    print("Benchmarking EKF on random acceleration track")
    average_score = 0
    for _ in range(N):
        measurements, trueTrack = simulateRandomAccelTrackPolar(
            v_x=1,
            v_y=1,
            x0=5,
            y0=5,
            num_datapoints=60,
            dt=dt,
            sigma_r=np.sqrt(var_r),
            sigma_phi=np.sqrt(var_phi),
        )
        # Define the kalman filter.
        KF = ExtendedKalmanFilter(f, h, F, H, Q, R, x0, P0)

        # Initialize the history array.
        x_history = np.zeros((6, len(measurements[0, :])))

        # Iterate over measurements to implement the recursive structure.
        for i in range(len(measurements[0, :])):
            KF.predict()
            KF.update(measurements[:, i].reshape(2, 1))

            x_history[:, i] = KF.x.reshape(
                6,
            )

        average_ospa = get_average_ospa(x_history, trueTrack)
        average_score += average_ospa
    print(
        f"Average OSPA for EKF = {average_score / N} over {N} random acceleration tracks"
    )


def BenchmarkUKF(f, h, Q, R, x0, P0, var_r, var_phi, dt, alpha, beta, kappa, N):
    print("Benchmarking UKF on linear track")
    average_score = 0
    failed = False
    for _ in range(N):
        measurements, trueTrack = simulateLinearTrackPolar(
            v_x=1,
            v_y=1,
            x0=5,
            y0=5,
            num_datapoints=60,
            dt=dt,
            sigma_r=np.sqrt(var_r),
            sigma_phi=np.sqrt(var_phi),
        )
        # Define the kalman filter.
        KF = UnscentedKalmanFilter(f, h, Q, R, x0, P0, alpha, beta, kappa)

        # Initialize the history array.
        x_history = np.zeros((6, len(measurements[0, :])))

        # Iterate over measurements to implement the recursive structure.
        try:
            for i in range(len(measurements[0, :])):
                KF.predict()
                KF.update(measurements[:, i].reshape(2, 1))

                x_history[:, i] = KF.x.reshape(
                    6,
                )

            average_ospa = get_average_ospa(x_history, trueTrack)
            average_score += average_ospa
        except np.linalg.LinAlgError:
            failed = True
            break
    if failed:
        print("UKF failed on at least 1 linear track")
    else:
        print(f"Average OSPA for UKF = {average_score / N} over {N} linear tracks")

    print("Benchmarking UKF on random acceleration track")
    average_score = 0
    failed = False
    for _ in range(N):
        measurements, trueTrack = simulateRandomAccelTrackPolar(
            v_x=1,
            v_y=1,
            x0=5,
            y0=5,
            num_datapoints=60,
            dt=dt,
            sigma_r=np.sqrt(var_r),
            sigma_phi=np.sqrt(var_phi),
        )
        # Define the kalman filter.
        KF = UnscentedKalmanFilter(f, h, Q, R, x0, P0, alpha, beta, kappa)

        # Initialize the history array.
        x_history = np.zeros((6, len(measurements[0, :])))

        # Iterate over measurements to implement the recursive structure.
        try:
            for i in range(len(measurements[0, :])):
                KF.predict()
                KF.update(measurements[:, i].reshape(2, 1))

                x_history[:, i] = KF.x.reshape(
                    6,
                )

            average_ospa = get_average_ospa(x_history, trueTrack)
            average_score += average_ospa
        except np.linalg.LinAlgError:
            failed = True
            break
    if failed:
        print("UKF failed on at least 1 random acceleration track")
    else:
        print(
            f"Average OSPA for UKF = {average_score / N} over {N} random acceleration tracks"
        )


def BenchmarkUKF_stonesoup(
    f,
    h,
    Q,
    R,
    x0,
    P0,
    alpha,
    beta,
    kappa,
    dt,
    start_time,
    shared_config,
    drones_params,
    N,
):
    print("Benchmarking UKF on random acceleration track")
    total_ospa = 0
    failed = False
    for _ in range(N):
        detections, groundTruths = SimulatorPolarMultitarget_stonesoup(
            **shared_config, drone_configs=drones_params
        )
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

        # Iterate over measurements to implement the recursive structure.
        try:
            for n, measurements in enumerate(detections):

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
                    measurements - associated_measurements,
                    start_time + timedelta(seconds=dt * n),
                )
                all_tracks |= tracks

            total_ospa += mean_ospa_stonesoup(
                track=all_tracks, ground_truth=groundTruths
            )
        except np.linalg.LinAlgError:
            # Penalize failed filter runs heavily so the optimizer avoids this region
            failed = True
            break
    if failed:
        print("UKF failed on at least 1 random acceleration track")
    else:
        print(
            f"Average OSPA for UKF = {total_ospa / N} over {N} random acceleration tracks"
        )


def BenchmarkJoint(
    f_matrix,
    f,
    F,
    h_matrix,
    h,
    H,
    Q_UCMKF,
    Q_EKF,
    Q_UKF,
    R_EKF,
    R_UKF,
    x0,
    P0_UCMKF,
    P0_EKF,
    P0_UKF,
    sigma_r,
    sigma_phi,
    dt,
    alpha,
    beta,
    kappa,
    N,
):
    print("Benchmarking UCMKF + EKF + UKF on linear track")
    average_score_UCMKF = 0
    average_score_EKF = 0
    average_score_UKF = 0
    failed = False
    for _ in range(N):
        measurements, trueTrack = simulateLinearTrackPolar(
            v_x=1,
            v_y=1,
            x0=5,
            y0=5,
            num_datapoints=60,
            dt=dt,
            sigma_r=sigma_r,
            sigma_phi=sigma_phi,
        )
        # Define the kalman filters.
        UCMKF = UCMKalmanFilter(
            f_matrix, h_matrix, Q_UCMKF, sigma_r, sigma_phi, x0, P0_UCMKF
        )
        EKF = ExtendedKalmanFilter(f, h, F, H, Q_EKF, R_EKF, x0, P0_EKF)
        UKF = UnscentedKalmanFilter(f, h, Q_UKF, R_UKF, x0, P0_UKF, alpha, beta, kappa)

        # Initialize the history arrays.
        x_history_UCMKF = np.zeros((6, len(measurements[0, :])))
        x_history_EKF = np.zeros((6, len(measurements[0, :])))
        x_history_UKF = np.zeros((6, len(measurements[0, :])))

        # Iterate over measurements to implement the recursive structure.
        try:
            for i in range(len(measurements[0, :])):
                UCMKF.predict()
                UCMKF.update(measurements[:, i].reshape(2, 1))

                x_history_UCMKF[:, i] = UCMKF.x.reshape(
                    6,
                )

                EKF.predict()
                EKF.update(measurements[:, i].reshape(2, 1))

                x_history_EKF[:, i] = EKF.x.reshape(
                    6,
                )

                UKF.predict()
                UKF.update(measurements[:, i].reshape(2, 1))

                x_history_UKF[:, i] = UKF.x.reshape(
                    6,
                )

            average_ospa_UCMKF = get_average_ospa(x_history_UCMKF, trueTrack)
            average_score_UCMKF += average_ospa_UCMKF
            average_ospa_EKF = get_average_ospa(x_history_EKF, trueTrack)
            average_score_EKF += average_ospa_EKF
            average_ospa_UKF = get_average_ospa(x_history_UKF, trueTrack)
            average_score_UKF += average_ospa_UKF
        except np.linalg.LinAlgError:
            failed = True
            break
    if failed:
        print("UKF failed on at least 1 linear track")
    else:
        print(
            f"Average OSPA for UCMKF = {average_score_UCMKF / N} over {N} linear tracks"
        )
        print(f"Average OSPA for EKF = {average_score_EKF / N} over {N} linear tracks")
        print(f"Average OSPA for UKF = {average_score_UKF / N} over {N} linear tracks")

    print("Benchmarking UCMKF + EKF + UKF on random acceleration track")
    average_score_UCMKF = 0
    average_score_EKF = 0
    average_score_UKF = 0
    failed = False
    for _ in range(N):
        measurements, trueTrack = simulateRandomAccelTrackPolar(
            v_x=1,
            v_y=1,
            x0=5,
            y0=5,
            num_datapoints=60,
            dt=dt,
            sigma_r=sigma_r,
            sigma_phi=sigma_phi,
        )
        # Define the kalman filters.
        UCMKF = UCMKalmanFilter(
            f_matrix, h_matrix, Q_UCMKF, sigma_r, sigma_phi, x0, P0_UCMKF
        )
        EKF = ExtendedKalmanFilter(f, h, F, H, Q_EKF, R_EKF, x0, P0_EKF)
        UKF = UnscentedKalmanFilter(f, h, Q_UKF, R_UKF, x0, P0_UKF, alpha, beta, kappa)

        # Initialize the history arrays.
        x_history_UCMKF = np.zeros((6, len(measurements[0, :])))
        x_history_EKF = np.zeros((6, len(measurements[0, :])))
        x_history_UKF = np.zeros((6, len(measurements[0, :])))

        # Iterate over measurements to implement the recursive structure.
        try:
            for i in range(len(measurements[0, :])):
                UCMKF.predict()
                UCMKF.update(measurements[:, i].reshape(2, 1))

                x_history_UCMKF[:, i] = UCMKF.x.reshape(
                    6,
                )

                EKF.predict()
                EKF.update(measurements[:, i].reshape(2, 1))

                x_history_EKF[:, i] = EKF.x.reshape(
                    6,
                )

                UKF.predict()
                UKF.update(measurements[:, i].reshape(2, 1))

                x_history_UKF[:, i] = UKF.x.reshape(
                    6,
                )

            average_ospa_UCMKF = get_average_ospa(x_history_UCMKF, trueTrack)
            average_score_UCMKF += average_ospa_UCMKF
            average_ospa_EKF = get_average_ospa(x_history_EKF, trueTrack)
            average_score_EKF += average_ospa_EKF
            average_ospa_UKF = get_average_ospa(x_history_UKF, trueTrack)
            average_score_UKF += average_ospa_UKF
        except np.linalg.LinAlgError:
            failed = True
            break
    if failed:
        print("UKF failed on at least 1 random acceleration track")
    else:
        print(
            f"Average OSPA for UCMKF = {average_score_UCMKF / N} over {N} random acceleration tracks"
        )
        print(
            f"Average OSPA for EKF = {average_score_EKF / N} over {N} random acceleration tracks"
        )
        print(
            f"Average OSPA for UKF = {average_score_UKF / N} over {N} random acceleration tracks"
        )
