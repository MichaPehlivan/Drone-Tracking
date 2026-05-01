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
    measurements = np.zeros((2,num_datapoints))

    for i in range(num_datapoints):
        measurements[0,i] = x + v_x*t + sigma*randn()
        measurements[1,i] = y + v_y*t + sigma*randn()
        t += dt

    return measurements