# Imports
import numpy as np
from numpy.random import randn

"""
simulateLinearTrack simulates a linear path taken by a drone

input:
    v_x: initial velocity in x-direction. (m/s)
    v_y: initial velocity in y-direction. (m/s)
    x0: initial x-position. (m)
    y0: initial y-position. (m)
    num_datapoints: number of datapoints in the track.
    dt: timestep (s).
    sigma: standard deviation of the measurement noise.
    
output:
    measurements: [2 x num_datapoints] array 
                  (x,y coordinates as rows, samples as columns)
                  
                  containing simulated measurements for a straight-line drone track.
"""


def simulateLinearTrack(v_x, v_y, x0, y0, num_datapoints, dt, sigma):
    x = x0
    y = y0
    t = 0
    trueTrack = np.zeros((2, num_datapoints))
    measurements = np.zeros((2, num_datapoints))

    for i in range(num_datapoints):
        trueTrack[0, i] = x + v_x * t
        trueTrack[1, i] = y + v_y * t

        measurements[0, i] = trueTrack[0, i] + sigma * randn()
        measurements[1, i] = trueTrack[1, i] + sigma * randn()

        t += dt

    return measurements, trueTrack


def simulateLinearTrackPolar(v_x, v_y, x0, y0, num_datapoints, dt, sigma_r, sigma_phi):
    x = x0
    y = y0
    t = 0
    trueTrack = np.zeros((2, num_datapoints))
    measurements = np.zeros((2, num_datapoints))

    for i in range(num_datapoints):
        trueTrack[0, i] = x + v_x * t
        trueTrack[1, i] = y + v_y * t

        measurements[0, i] = (
            np.sqrt(trueTrack[0, i] ** 2 + trueTrack[1, i] ** 2) + sigma_r * randn()
        )  # range
        measurements[1, i] = (
            np.arctan2(trueTrack[1, i], trueTrack[0, i]) + sigma_phi * randn()
        )  # azimuth

        t += dt

    return measurements, trueTrack
