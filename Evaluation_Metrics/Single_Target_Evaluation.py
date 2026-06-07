import numpy as np
import matplotlib.pyplot as plt

# Reference (GOSPA): A. S. Rahmathullah, Á. F. García-Fernández and L. Svensson, "Generalized optimal sub-pattern assignment metric,"
#            2017 20th International Conference on Information Fusion (Fusion), Xi'an, China, 2017, pp. 1-8, doi: 10.23919/ICIF.2017.8009645.

# Reference (Multitarget OSPA): B. Ristic, B. -N. Vo, D. Clark and B. -T. Vo, "A Metric for Performance Evaluation of Multi-Target Tracking Algorithms,"
#               in IEEE Transactions on Signal Processing, vol. 59, no. 7, pp. 3452-3457, July 2011, doi: 10.1109/TSP.2011.2140111
#
# ______________

from stonesoup.metricgenerator.ospametric import OSPAMetric
from stonesoup.types.state import State
from stonesoup.types.array import StateVector
from stonesoup.dataassociator.tracktotrack import TrackToTruth
from stonesoup.metricgenerator.manager import SimpleManager
from stonesoup.measures import Euclidean

"""
This function calculates the average OSPA distance for a single track using the stonesoup library.

"""


def get_average_ospa(x_history, trueTrack):

    ospa_calc = OSPAMetric(p=2, c=10.0)
    ospa_values = []

    for k in range(x_history.shape[1]):

        track_state = [State(state_vector=StateVector(x_history[:2, k]))]
        true_state = [State(state_vector=StateVector(trueTrack[:2, k]))]

        distance = ospa_calc.compute_OSPA_distance(track_state, true_state)
        ospa_values.append(distance.value)

    average_ospa = np.mean(ospa_values)

    return average_ospa


def ospa_stonesoup(track, ground_truth, c=10, p=1):

    pos_measure = Euclidean(mapping=[0, 3])

    ospa_generator = OSPAMetric(c=c, p=p, measure=pos_measure, generator_name="OSPA")

    associator = TrackToTruth(association_threshold=15, measure=pos_measure)

    metric_manager = SimpleManager([ospa_generator], associator=associator)

    metric_manager.add_data(ground_truth, track)

    metrics = metric_manager.generate_metrics()
    ospa_metric = metrics["OSPA distances"]
    # print(f"Available metrics: {metrics.keys()}")

    ospa_times = [metric.timestamp for metric in ospa_metric.value]
    ospa_values = [metric.value for metric in ospa_metric.value]

    # Convert timestamps to relative execution seconds for a clean X-axis scale

    # Plot using matplotlib
    plt.figure(figsize=(8, 4))
    plt.plot(
        ospa_times, ospa_values, color="crimson", linewidth=2, label="OSPA Distance"
    )
    plt.axhline(
        np.mean(ospa_values),
        color="gray",
        linestyle="--",
        label=f"Mean OSPA ({np.mean(ospa_values):.2f}m)",
    )
    ospa_values_in_frame = [x for x in ospa_values if x < max(ospa_values)]
    plt.axhline(
        np.mean(ospa_values_in_frame),
        color="blue",
        linestyle="--",
        label=f"Mean OSPA in frame ({np.mean(ospa_values_in_frame):.2f}m)",
    )

    plt.title("OSPA Metric Evaluation Over Time", fontsize=11, fontweight="bold")
    plt.xlabel("Simulation Time (seconds)")
    plt.ylabel("OSPA (meters)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.show()


def mean_ospa_stonesoup(track, ground_truth, c=10, p=1, in_frame=False):
    pos_measure = Euclidean(mapping=[0, 3])

    ospa_generator = OSPAMetric(c=c, p=p, measure=pos_measure, generator_name="OSPA")

    associator = TrackToTruth(association_threshold=15, measure=pos_measure)

    metric_manager = SimpleManager([ospa_generator], associator=associator)

    metric_manager.add_data(ground_truth, track)

    metrics = metric_manager.generate_metrics()
    ospa_metric = metrics["OSPA distances"]

    ospa_values = [metric.value for metric in ospa_metric.value]
    if in_frame:
        ospa_values_in_frame = [x for x in ospa_values if x < max(ospa_values)]
        return np.mean(ospa_values_in_frame)
    else:
        return np.mean(ospa_values)
