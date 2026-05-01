import numpy as np
"""
generateErrorMatrix generates an error matrix P for the simple kalman filter implementation.

"""

def generateErrorMatrix(c):
    P0  =   c*np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    return P0