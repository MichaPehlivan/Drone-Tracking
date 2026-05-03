import numpy as np

# Reference (OSPA): A. S. Rahmathullah, Á. F. García-Fernández and L. Svensson, "Generalized optimal sub-pattern assignment metric,"
#            2017 20th International Conference on Information Fusion (Fusion), Xi'an, China, 2017, pp. 1-8, doi: 10.23919/ICIF.2017.8009645.

# Reference (Multitarget OSPA): B. Ristic, B. -N. Vo, D. Clark and B. -T. Vo, "A Metric for Performance Evaluation of Multi-Target Tracking Algorithms,"
#               in IEEE Transactions on Signal Processing, vol. 59, no. 7, pp. 3452-3457, July 2011, doi: 10.1109/TSP.2011.2140111
#
# ______________

from stonesoup.metricgenerator.ospametric import OSPAMetric
from stonesoup.types.state import State
from stonesoup.types.array import StateVector

"""
This function calculates the average OSPA distance for a single track using the stonesoup library.

"""
def get_average_ospa(x_history, trueTrack):

    ospa_calc = OSPAMetric(p=1, c=10.0)
    ospa_values = []

    # Loop for k = 1, ..., K as shown in image_edc5bb.png
    for k in range(x_history.shape[1]):
        # Form labeled sets (Line 8)
        track_state = [State(state_vector=StateVector(x_history[:2, k]))]
        true_state = [State(state_vector=StateVector(trueTrack[:2, k]))]

        # Compute distance at t_k (Line 9)
        distance = ospa_calc.compute_OSPA_distance(track_state, true_state)
        ospa_values.append(distance.value)

    average_ospa = np.mean(ospa_values)


    return average_ospa
