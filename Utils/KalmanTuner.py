import numpy as np

import optuna

from Evaluation_Metrics.Single_Target_Evaluation import get_average_ospa
from Kalman_Filters import UCMKalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from Track_Simulation import simulateRandomAccelTrackPolar


def TuneEKF(f, h, F, H, x0, var_r, var_phi, dt, N):
    Q_base = 1.0 * np.array(
        [
            [(dt**5) / 20, 0, (dt**4) / 8, 0, (dt**3) / 6, 0],
            [0, (dt**5) / 20, 0, (dt**4) / 8, 0, (dt**3) / 6],
            [(dt**4) / 8, 0, (dt**3) / 3, 0, (dt**2) / 2, 0],
            [0, (dt**4) / 8, 0, (dt**3) / 3, 0, (dt**2) / 2],
            [(dt**3) / 6, 0, (dt**2) / 2, 0, dt, 0],
            [0, (dt**3) / 6, 0, (dt**2) / 2, 0, dt],
        ]
    )

    R_base = 1 * np.array([[var_r, 0], [0, var_phi]])

    P0_base = 1 * np.eye(6)

    Q_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    R_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    P_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])

    eval_data = []
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
        eval_data.append((measurements, trueTrack))

    scores = {}

    for Q_mul in Q_space:
        print(f"|Q| = {Q_mul}")
        for R_mul in R_space:
            for P0_mul in P_space:
                Q = Q_base * Q_mul
                R = R_base * R_mul
                P0 = P0_base * P0_mul
                avarage_score = 0
                for measurements, trueTrack in eval_data:
                    # Define the kalman filter.
                    KF = ExtendedKalmanFilter(f, h, F, H, Q, R, P0, x0)

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


def TuneUKF(f, h, x0, var_r, var_phi, dt, beta, kappa, N):
    Q_base = 1.0 * np.array(
        [
            [(dt**5) / 20, 0, (dt**4) / 8, 0, (dt**3) / 6, 0],
            [0, (dt**5) / 20, 0, (dt**4) / 8, 0, (dt**3) / 6],
            [(dt**4) / 8, 0, (dt**3) / 3, 0, (dt**2) / 2, 0],
            [0, (dt**4) / 8, 0, (dt**3) / 3, 0, (dt**2) / 2],
            [(dt**3) / 6, 0, (dt**2) / 2, 0, dt, 0],
            [0, (dt**3) / 6, 0, (dt**2) / 2, 0, dt],
        ]
    )

    R_base = 1 * np.array([[var_r, 0], [0, var_phi]])

    P0_base = 1 * np.eye(6)

    Q_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    R_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    P_space = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    alpha_space = np.array([1e-3, 1e-2, 0.1, 0.5, 1, 2])

    eval_data = []
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
        eval_data.append((measurements, trueTrack))

    scores = {}

    for Q_mul in Q_space:
        print(f"|Q| = {Q_mul}")
        for R_mul in R_space:
            for P0_mul in P_space:
                for alpha in alpha_space:
                    Q = Q_base * Q_mul
                    R = R_base * R_mul
                    P0 = P0_base * P0_mul
                    average_score = 0
                    failed = False
                    for measurements, trueTrack in eval_data:
                        # Define the kalman filter.
                        KF = UnscentedKalmanFilter(
                            f, h, Q, R, P0, alpha, beta, kappa, x0
                        )

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
                    if not failed:
                        scores[
                            f"|Q|={Q_mul}, |R|={R_mul}, |P0|={P0_mul}, alpha={alpha}"
                        ] = (average_score / N)
                    else:
                        scores[
                            f"|Q|={Q_mul}, |R|={R_mul}, |P0|={P0_mul}, alpha={alpha}"
                        ] = 0

    min_params = min((k for k in scores if scores[k] != 0), key=lambda k: scores[k])
    print(f"Minimum OSPA for UKF = {scores[min_params]}, with {min_params}")


def optimize_UCMKF(f, h, x0, sigma_r, sigma_phi, dt, N):
    Q_base = 1.0 * np.array(
        [
            [(dt**5) / 20, 0, (dt**4) / 8, 0, (dt**3) / 6, 0],
            [0, (dt**5) / 20, 0, (dt**4) / 8, 0, (dt**3) / 6],
            [(dt**4) / 8, 0, (dt**3) / 3, 0, (dt**2) / 2, 0],
            [0, (dt**4) / 8, 0, (dt**3) / 3, 0, (dt**2) / 2],
            [(dt**3) / 6, 0, (dt**2) / 2, 0, dt, 0],
            [0, (dt**3) / 6, 0, (dt**2) / 2, 0, dt],
        ]
    )

    P0_base = 1 * np.eye(6)

    evaluation_data = []
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
        evaluation_data.append((measurements, trueTrack))

    def objective(trial):
        # Sample parameters continuously on a log scale
        Q_mul = trial.suggest_float("|Q|", 1e-4, 1e3, log=True)
        P0_mul = trial.suggest_float("|P0|", 1e-4, 1e3, log=True)

        Q = Q_base * Q_mul
        P0 = P0_base * P0_mul

        total_ospa = 0
        for measurements, trueTrack in evaluation_data:
            KF = UCMKalmanFilter(f, h, Q, P0, sigma_r, sigma_phi, x0)
            x_history = np.zeros((6, len(measurements[0, :])))

            try:
                for i in range(len(measurements[0, :])):
                    KF.predict()
                    KF.update(measurements[:, i].reshape(2, 1))
                    x_history[:, i] = KF.x.reshape(
                        6,
                    )

                total_ospa += get_average_ospa(x_history, trueTrack)
            except np.linalg.LinAlgError:
                # Penalize failed filter runs heavily so the optimizer avoids this region
                return float("inf")

        return total_ospa / len(evaluation_data)

    # Create and run the study
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective, n_trials=150
    )  # 100 intelligent guesses instead of 512 blind ones

    print(f"Minimum OSPA for UCMKF = {study.best_value}, with {study.best_params}")


def optimize_EKF(f, h, F, H, x0, var_r, var_phi, dt, N):
    Q_base = 1.0 * np.array(
        [
            [(dt**5) / 20, 0, (dt**4) / 8, 0, (dt**3) / 6, 0],
            [0, (dt**5) / 20, 0, (dt**4) / 8, 0, (dt**3) / 6],
            [(dt**4) / 8, 0, (dt**3) / 3, 0, (dt**2) / 2, 0],
            [0, (dt**4) / 8, 0, (dt**3) / 3, 0, (dt**2) / 2],
            [(dt**3) / 6, 0, (dt**2) / 2, 0, dt, 0],
            [0, (dt**3) / 6, 0, (dt**2) / 2, 0, dt],
        ]
    )

    R_base = 1 * np.array([[var_r, 0], [0, var_phi]])

    P0_base = 1 * np.eye(6)

    evaluation_data = []
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
        evaluation_data.append((measurements, trueTrack))

    def objective(trial):
        # Sample parameters continuously on a log scale
        Q_mul = trial.suggest_float("|Q|", 1e-4, 1e3, log=True)
        R_mul = trial.suggest_float("|R|", 1e-4, 1e3, log=True)
        P0_mul = trial.suggest_float("|P0|", 1e-4, 1e3, log=True)

        Q = Q_base * Q_mul
        R = R_base * R_mul
        P0 = P0_base * P0_mul

        total_ospa = 0
        for measurements, trueTrack in evaluation_data:
            KF = ExtendedKalmanFilter(f, h, F, H, Q, R, P0, x0)
            x_history = np.zeros((6, len(measurements[0, :])))

            try:
                for i in range(len(measurements[0, :])):
                    KF.predict()
                    KF.update(measurements[:, i].reshape(2, 1))
                    x_history[:, i] = KF.x.reshape(
                        6,
                    )

                total_ospa += get_average_ospa(x_history, trueTrack)
            except np.linalg.LinAlgError:
                # Penalize failed filter runs heavily so the optimizer avoids this region
                return float("inf")

        return total_ospa / len(evaluation_data)

    # Create and run the study
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective, n_trials=150
    )  # 100 intelligent guesses instead of 512 blind ones

    print(f"Minimum OSPA for EKF = {study.best_value}, with {study.best_params}")


def optimize_UKF(f, h, x0, beta, kappa, var_r, var_phi, dt, N):
    Q_base = 1.0 * np.array(
        [
            [(dt**5) / 20, 0, (dt**4) / 8, 0, (dt**3) / 6, 0],
            [0, (dt**5) / 20, 0, (dt**4) / 8, 0, (dt**3) / 6],
            [(dt**4) / 8, 0, (dt**3) / 3, 0, (dt**2) / 2, 0],
            [0, (dt**4) / 8, 0, (dt**3) / 3, 0, (dt**2) / 2],
            [(dt**3) / 6, 0, (dt**2) / 2, 0, dt, 0],
            [0, (dt**3) / 6, 0, (dt**2) / 2, 0, dt],
        ]
    )

    R_base = 1 * np.array([[var_r, 0], [0, var_phi]])

    P0_base = 1 * np.eye(6)

    evaluation_data = []
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
        evaluation_data.append((measurements, trueTrack))

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
        for measurements, trueTrack in evaluation_data:
            # Define the kalman filter.
            KF = UnscentedKalmanFilter(f, h, Q, R, P0, alpha, beta, kappa, x0)

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

                total_ospa += get_average_ospa(x_history, trueTrack)
            except np.linalg.LinAlgError:
                # Penalize failed filter runs heavily so the optimizer avoids this region
                return float("inf")

        return total_ospa / len(evaluation_data)

    # Create and run the study
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective, n_trials=150
    )  # 100 intelligent guesses instead of 512 blind ones

    print(f"Minimum OSPA for UKF = {study.best_value}, with {study.best_params}")
