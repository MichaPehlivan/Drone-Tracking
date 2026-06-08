# External Imports
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from pipeline import run_algorithm


if __name__ == '__main__':

    runtime = []

    # for i in range(50):
    #
    #     (runtime_i, ospa_values, ospa_corrected_values) = run_algorithm(
    #
    #         filter = "ucmkf" , # "ucmkf", "ekf", or "ukf"
    #         model = "ca",  # "cv" or "ca"
    #         ndoppler = 500,
    #         recording_path = "Data/sidetoside/detections_mvdr_argmax_10db_500.csv",
    #         gps_path= "Data/sidetoside/flight2-sidetoside-GPS.csv",
    #         association_distance = 4,
    #         deletion_covariance = 15,
    #         initiation_points = 15,
    #         gps_offset_delay =  40 #offset for the recordings
    #     )
    #     runtime.append(runtime_i)
    #
    # print(f"final runtime over 50 iterations: {np.mean(runtime):.4f}")

    #sidetoside
    (runtime_i, ospa_values, ospa_corrected_values) = run_algorithm(

            filter = "ucmkf" , # "ucmkf", "ekf", or "ukf"
            model = "ca",  # "cv" or "ca"
            ndoppler = 500,
            recording_path = "Data/sidetoside/detections_mvdr_argmax_10db_500.csv",
            gps_path= "Data/sidetoside/flight2-sidetoside-GPS.csv",
            association_distance = 4,
            deletion_covariance = 15,
            initiation_points = 15,
            gps_offset_delay =  40 #offset for the recordings
            )

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

    #HOVERING
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
