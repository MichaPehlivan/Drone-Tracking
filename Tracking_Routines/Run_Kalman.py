# External Imports
import numpy as np

# Internal
from Kalman_Filters import KalmanFilter, ExtendedKalmanFilter
from Utils import plotSimpleKalman
from Evaluation_Metrics import get_average_ospa

"""
This performs a simple kalman filter on simulated data.
inputs are the same as for the kalman filter + the necessary measurements
"""


def RunSimpleKalman(F, H, Q, R, x0, P0, measurements, trueTrack):

    # Define the kalman filter.
    KF = KalmanFilter(F, H, Q, R, x0, P0)

    # Initialize the history array.
    x_history = np.zeros((4, len(measurements[0, :])))

    # Iterate over measurements to implement the recursive structure.
    for i in range(len(measurements[0, :])):
        KF.predict()
        KF.update(measurements[:, i].reshape(2, 1))

        x_history[:, i] = KF.x.reshape(
            4,
        )

    average_ospa = get_average_ospa(x_history, trueTrack)

    # Uses the plotting module to plot the x_history.
    plotSimpleKalman(x_history, measurements, trueTrack, average_ospa)

    return


"""
This performs an extended kalman filter on simulated data.
inputs are the same as for the kalman filter + the necessary measurements
"""


def RunExtendedKalman(f, h, F, H, Q, R, x0, P0, measurements, trueTrack, polar=True):

    # Define the kalman filter.
    KF = ExtendedKalmanFilter(f, h, F, H, Q, R, x0, P0)

    # Initialize the history array.
    x_history = np.zeros((4, len(measurements[0, :])))

    # Iterate over measurements to implement the recursive structure.
    for i in range(len(measurements[0, :])):
        KF.predict()
        KF.update(measurements[:, i].reshape(2, 1))

        x_history[:, i] = KF.x.reshape(
            4,
        )

    average_ospa = get_average_ospa(x_history, trueTrack)

    plot_measurements = np.zeros_like(measurements)
    plot_measurements[0, :] = measurements[0, :] * np.cos(
        measurements[1, :]
    )  # x = r * cos(theta)
    plot_measurements[1, :] = measurements[0, :] * np.sin(
        measurements[1, :]
    )  # y = r * sin(theta)



    # Uses the plotting module to plot the x_history.
    if polar:
        plotSimpleKalman(x_history, plot_measurements, trueTrack, average_ospa)

    else:
        plotSimpleKalman(x_history, measurements, trueTrack, average_ospa)

    return
