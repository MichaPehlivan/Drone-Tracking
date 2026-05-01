# External Imports
import numpy as np

# Internal
from Kalman_Filters import KalmanFilter
from Utils import plotSimpleKalman

"""
This performs a simple kalman filter on simulated data.
inputs are the same as for the kalman filter + the necessary measurements
"""

def RunSimpleKalman(F, H, Q, R, x0, P0, measurements):

    #Define the kalman filter.
    KF = KalmanFilter(F, H, Q, R, x0, P0)

    #Initialize the history array.
    x_history = np.zeros((4, len(measurements[0,:]) ))

    #Iterate over measurements to implement the recursive structure.
    for i in range(len(measurements[0,:])):
        KF.predict()
        KF.update(measurements[:, i].reshape(2, 1))

        x_history[:, i] = KF.x.reshape(4, )

    #Uses the plotting module to plot the x_history.
    plotSimpleKalman(x_history, measurements)

    return