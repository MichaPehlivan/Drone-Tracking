import numpy as np

from Evaluation_Metrics.Single_Target_Evaluation import get_average_ospa
from Kalman_Filters import ExtendedKalmanFilter, UnscentedKalmanFilter


def TuneEKF(f, h, F, H, x0, var_r, var_phi, measurements, trueTrack, N):
    Q_base = 1.0 * np.eye(6)

    R_base = 1 * np.array([[var_r, 0], [0, var_phi]])

    P0_base = 1 * np.eye(6)

    Q_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    R_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    P_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])

    scores = {}

    for Q_mul in Q_space:
        for R_mul in R_space:
            for P0_mul in P_space:
                Q = Q_base * Q_mul
                R = R_base * R_mul
                P0 = P0_base * P0_mul
                avarage_score = 0
                for _ in range(N):
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
                    avarage_score += average_ospa
                scores[f"|Q|={Q_mul}, |R|={R_mul}, |P0|={P0_mul}"] = avarage_score / N

    min_params = min(scores, key=scores.get)
    print(f"Minimum OSPA for EKF = {scores[min_params]}, with {min_params}")


def TuneUKF(f, h, x0, var_r, var_phi, beta, kappa, measurements, trueTrack, N):
    Q_base = 1.0 * np.eye(6)

    R_base = 1 * np.array([[var_r, 0], [0, var_phi]])

    P0_base = 1 * np.eye(6)

    Q_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    R_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    P_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    alpha_space = np.array([1e-3, 1e-2, 0.1, 0.5, 1, 2])

    scores = {}

    for Q_mul in Q_space:
        for R_mul in R_space:
            for P0_mul in P_space:
                for alpha in alpha_space:
                    Q = Q_base * Q_mul
                    R = R_base * R_mul
                    P0 = P0_base * P0_mul
                    average_score = 0
                    for _ in range(N):
                        # Define the kalman filter.
                        KF = UnscentedKalmanFilter(
                            f, h, Q, R, x0, P0, alpha, beta, kappa
                        )

                        # Initialize the history array.
                        x_history = np.zeros((6, len(measurements[0, :])))

                        # Iterate over measurements to implement the recursive structure.
                        try:
                            for i in range(len(measurements[0, :])):
                                _, sigma = KF.predict()
                                KF.update(sigma, measurements[:, i].reshape(2, 1))

                                x_history[:, i] = KF.x.reshape(
                                    6,
                                )

                            average_ospa = get_average_ospa(x_history, trueTrack)
                            average_score += average_ospa
                        except np.linalg.LinAlgError:
                            continue
                    scores[
                        f"|Q|={Q_mul}, |R|={R_mul}, |P0|={P0_mul}, alpha={alpha}"
                    ] = (average_score / N)

    min_params = min((k for k in scores if scores[k] != 0), key=lambda k: scores[k])
    print(f"Minimum OSPA for UKF = {scores[min_params]}, with {min_params}")
