# External Imports
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from timeit import default_timer as timer
# Internal Packages

from Evaluation_Metrics import ospa_stonesoup
from Kalman_Filters import UCMKalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from Utils.DecodeGPS import gps_to_ground_truth, interpolate_ground_truth
from Utils.Wrapper_Functions import (
    UCMKFPredictor,
    UCMKFUpdater,
    EKFPredictor,
    EKFUpdater,
    UKFPredictor,
    UKFUpdater,
)
from Utils import ReadDetections, ReadAndClusterDetections

# Stonesoup imports
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.state import GaussianState
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.plotter import Plotter



def run_algorithm(filter, model, ndoppler, recording_path, gps_path, association_distance = 4 ,deletion_covariance = 15 ,initiation_points = 15, gps_offset_delay=0):

    # print("Starting Runtime")

    # variances in measurement dimensions.
    range_sigma = np.sqrt(0.444)
    azimuth_sigma = np.sqrt(0.000720)

    # Kalman Filter tuning
    UCMKF_Q = 1
    UCMKF_P0 = 1
    EKF_Q = 1
    EKF_R = 1
    EKF_P0 = 1
    UKF_Q = 1
    UKF_R = 1
    UKF_P0 = 1
    alpha = 1

    dt = 210e-6 * ndoppler


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
                    (-x[2]) / (1e-9+x[0] ** 2 + x[2] ** 2),
                    0,
                    x[0] / (1e-9+x[0] ** 2 + x[2] ** 2 ),
                    0,
                ],
                [
                    x[0] / np.sqrt(1e-9+x[0] ** 2 + x[2] ** 2),
                    0,
                    x[2] / np.sqrt(1e-9+x[0] ** 2 + x[2] ** 2),
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
            [[Q1D_generator(dt), np.zeros((2, 2))], [np.zeros((2, 2)), Q1D_generator(dt)]]
        )
        if model == "cv"
        else np.block(
            [[Q1D_generator(dt), np.zeros((3, 3))], [np.zeros((3, 3)), Q1D_generator(dt)]]
        )
    )

    Q = Q_generator(dt)

    x0 = (
        np.array([[1], [0], [1], [0]])
        if model == "cv"
        else np.array([[1], [0], [0], [1], [0], [0]])
    )

    # Starting error covariance (should be on the higher side to quickly settle in towards the correct values. i.e high uncertainty to start with :))
    P0 = np.eye(4) if model == "cv" else np.eye(6)

    # convert to variance.
    var_r = 0.444 #Experimentally obtained values
    var_phi =  0.000720 #Also experimental (Rad)

    # Initialize measurement error matrix.
    R = np.array([[var_phi, 0], [0, var_r]])

    # Define measurement_model
    measurement_model = (
        CartesianToBearingRange(ndim_state=4, mapping=(0, 2), noise_covar=R)
        if model == "cv"
        else CartesianToBearingRange(ndim_state=6, mapping=(0, 3), noise_covar=R)
    )

    # start the clock for easy timestamp management
    start_time = datetime(2026, 5, 28, 8, 10, 18, 33)

    # Get the detections from Abdullahs group
    detections = ReadDetections(
        filepath=recording_path,
        measurement_model=measurement_model,
        dt=dt,
        start_time=start_time,
    )
    time_duration_recording_s = timedelta(seconds=dt) * len(detections)



    ground_truth = gps_to_ground_truth(
        filepath=gps_path, model=model, start_time=start_time, gps_offset_delay=gps_offset_delay
    )

    # ___Initialize the filter____
    kf = (
        UCMKalmanFilter(
            F=F,
            H=h,
            Q=UCMKF_Q * Q,
            P0=UCMKF_P0 * P0,
            sigma_r=range_sigma,
            sigma_phi=azimuth_sigma,
            x0=x0,
        )
        if filter == "ucmkf"
        else (
            ExtendedKalmanFilter(
                f=f, h=h, F=F, H=H, Q=EKF_Q * Q, R=EKF_R * R, P0=EKF_P0 * P0, x0=x0
            )
            if filter == "ekf"
            else UnscentedKalmanFilter(
                f=f,
                h=h,
                Q=UKF_Q * Q,
                R=UKF_R * R,
                P0=UKF_P0 * P0,
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
        predictor, updater, measure=Mahalanobis(), missed_distance=association_distance
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


    tracks, all_tracks = set(), set()
    timesteps = []
    previous_timestamp = None
    start = timer()
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
        tracks |= initiator.initiate(measurements - associated_measurements, timestamp)
        all_tracks |= tracks

    end = timer()

    duration = end - start
    # print(f"runtime: {duration} s")
    interpolated_ground_truth = interpolate_ground_truth(ground_truth[0], timesteps, model)

    # Evaluate using the ALIGNED ground truth
    OSPA_values, OSPA_corrected_values = ospa_stonesoup(all_tracks, interpolated_ground_truth)



    # plotter = Plotter()
    # plotter.plot_ground_truths(ground_truth, [0, 2] if model == "cv" else [0, 3])
    # plotter.plot_tracks(all_tracks, [0, 2] if model == "cv" else [0, 3])
    # plotter.plot_measurements(
    #     [det for det_set in detections for det in det_set],
    #     [0, 2] if model == "cv" else [0, 3],
    #     measurement_model=measurement_model,
    # )
    #
    # ax = plt.gca()
    # x_min, x_max = ax.get_xlim()
    # y_min, y_max = ax.get_ylim()
    # data_min = min(x_min, y_min)
    # data_max = max(x_max, y_max)
    # # ax.set_xlim(data_min, data_max)
    # # ax.set_ylim(data_min, data_max)
    # ax.set_xlim(-10, 50)
    # ax.set_ylim(-30, 30)
    # ax.set_aspect("equal", adjustable="box")
    #
    # plt.grid()
    # plotter.fig.show()
    # plt.show()
    from stonesoup.plotter import AnimatedPlotterly

    plotter = AnimatedPlotterly(timesteps, tail_length=0.7)
    plotter.fig.update_layout(
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        )
    )
    from stonesoup.sensor.radar.radar import RadarBearingRange
    from stonesoup.types.array import StateVector

    plotter.fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="x", yref="y",
                x0=-0.2, y0=-0.2, x1=0.2, y1=0.2,  # Draws a small circle around (0,0)
                fillcolor="blue",
                line=dict(color="black", width=0),
            )
        ]
    )

    plotter.fig.update_layout(width=700, height=700)
    plotter.plot_measurements(detections, [0, 2] if model == "cv" else [0, 3])
    plotter.plot_ground_truths(ground_truth, mapping=[0, 2] if model == "cv" else [0, 3])
    plotter.plot_tracks(all_tracks, [0, 2] if model == "cv" else [0, 3], uncertainty=False)


    plotter.fig.show()

    return duration, OSPA_values, OSPA_corrected_values