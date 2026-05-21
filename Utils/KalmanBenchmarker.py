import numpy as np

from Evaluation_Metrics.Single_Target_Evaluation import get_average_ospa
from Kalman_Filters import ExtendedKalmanFilter, UnscentedKalmanFilter
from Track_Simulation import simulateRandomAccelTrackPolar, simulateLinearTrackPolar


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
