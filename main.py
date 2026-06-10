# External Imports
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

from pipeline import generate_config, run_algorithm

if __name__ == "__main__":

    # CONFIG
    # filter = "ukf"  # "ucmkf", "ekf", or "ukf"
    # model = "ca"  # "cv" or "ca"

    ndoppler = 500
    dt = 210e-6 * ndoppler
    start_time = datetime(2026, 5, 28, 8, 10, 18, 33)

    recording_path = "Data/sidetoside/detections_mvdr_argmax_10db_500.csv"
    gps_path = "Data/sidetoside/flight2-sidetoside-GPS.csv"

    # select simulation or real data
    simulation = True
    plot = False

    x_initial = 5
    y_initial = 5
    num_datapoints = 100

    var_r = 0.444
    var_phi = 0.000720

    association_distance = 5
    deletion_covariance = 15
    initiation_points = 15
    gps_offset_delay = 40  # offset for the recordings

    drones_params = [
        {"x0": x_initial, "y0": y_initial, "v_x": 5.0, "v_y": 5.0},
        {"x0": x_initial + 100, "y0": y_initial, "v_x": -5, "v_y": 5},
        {
            "x0": x_initial + 100,
            "y0": y_initial + 70,
            "v_x": -5,
            "v_y": 1,
            "delay_steps": 10,
        },
        {
            "x0": x_initial - 100,
            "y0": y_initial - 70,
            "v_x": -5,
            "v_y": 1,
            "delay_steps": 10,
        },
    ]

    # sidetoside
    for filter in ["ucmkf", "ekf", "ukf"]:
        ospa_times_cv = []
        ospa_times_ca = []
        ospa_values_cv = []
        ospa_values_ca = []
        ospa_corrected_values_cv = []
        ospa_corrected_values_ca = []
        for model in ["cv", "ca"]:
            print(f"running {filter} with {model} model")
            (
                detections,
                dt,
                data_associator,
                updater,
                deleter,
                initiator,
                ground_truth,
            ) = generate_config(
                var_r,
                var_phi,
                start_time,
                dt,
                model,
                filter,
                drones_params,
                num_datapoints,
                recording_path,
                gps_path,
                gps_offset_delay,
                association_distance,
                deletion_covariance,
                initiation_points,
                simulation=simulation,
            )
            runtime_i, ospa_times, ospa_values, ospa_corrected_values = run_algorithm(
                model,
                detections,
                start_time,
                dt,
                data_associator,
                updater,
                deleter,
                initiator,
                ground_truth,
                simulation=simulation,
                plot=plot,
            )

            if model == "cv":
                ospa_times_cv = ospa_times
                ospa_values_cv = ospa_values
                ospa_corrected_values_cv = ospa_corrected_values
            else:
                ospa_times_ca = ospa_times
                ospa_values_ca = ospa_values
                ospa_corrected_values_ca = ospa_corrected_values

            print(
                f"runtime: {runtime_i}s, average OSPA: {np.mean(ospa_values)}, average corrected OSPA: {np.mean(ospa_corrected_values)}"
            )

        plt.figure(figsize=(8, 4))
        plt.plot(
            ospa_times_cv,
            ospa_values_cv,
            color="crimson",
            linewidth=2,
            label=f"OSPA Distance CV",
        )
        plt.plot(
            ospa_times_ca,
            ospa_values_ca,
            color="purple",
            linewidth=2,
            label=f"OSPA Distance CA",
        )
        plt.axhline(
            np.mean(ospa_values_cv),
            color="gray",
            linestyle="--",
            label=f"Mean OSPA CV ({np.mean(ospa_values_cv):.3f}m)",
        )
        plt.axhline(
            np.mean(ospa_values_ca),
            color="black",
            linestyle="--",
            label=f"Mean OSPA CA ({np.mean(ospa_values_ca):.3f}m)",
        )
        plt.axhline(
            np.mean(ospa_corrected_values_cv),
            color="blue",
            linestyle="--",
            label=f"Corrected OSPA CV ({np.mean(ospa_corrected_values_cv):.3f}m)",
        )
        plt.axhline(
            np.mean(ospa_corrected_values_ca),
            color="green",
            linestyle="--",
            label=f"Corrected OSPA CA ({np.mean(ospa_corrected_values_ca):.3f}m)",
        )

        plt.title(f"OSPA Over Time {filter}", fontsize=11, fontweight="bold")
        plt.xlabel("Time [s]")
        plt.ylabel("OSPA [m]")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.show()
    # (runtime_i, ospa_values, ospa_corrected_values) = run_algorithm(
    #
    #     filter="ukf",  # "ucmkf", "ekf", or "ukf"
    #     model="cv",  # "cv" or "ca"
    #     ndoppler=250,
    #     recording_path="Data/DronePerson/flight5-drone+person_tiny_tinyrad_master_1_USE_BG_True_ANGLE_cfar_rd_guided_music_range_angle_velocity_detections_ndoppler250.csv",
    #     gps_path="Data/DronePerson/May-28th-2026-10-12AM-Flight-Airdata.csv",
    #     association_distance=4,
    #     deletion_covariance=15,
    #     initiation_points=15,
    #     gps_offset_delay=40  # offset for the recordings
    #
    # )

    # HOVERING
    # (runtime_i, ospa_values, ospa_corrected_values) = run_algorithm(
    #
    #     filter="ukf",  # "ucmkf", "ekf", or "ukf"
    #     model="cv",  # "cv" or "ca"
    #     ndoppler=500,
    #     recording_path="Data/Hovering/flight6-hovering_tinyrad_master_1-ndoppler.csv",
    #     gps_path="Data/Hovering/flight6-hovering-GPS.csv",
    #     association_distance=4,
    #     deletion_covariance=15,
    #     initiation_points=15,
    #     gps_offset_delay=-3  # offset for the recordings
    #
    # )
