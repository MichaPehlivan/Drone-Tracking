from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# for the gps transformation, coordinate reference system en transformer -> reference is bij de radar (0,0)
from pyproj import CRS, Transformer

from stonesoup.types.state import State
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState


def gps_to_ground_truth(**kwargs):

    SensorLat = 51.98880548
    SensorLon = 4.390007015

    wgs_84 = CRS("EPSG:4326")
    sensor_position = CRS(
        f"+proj=ortho +lat_0={SensorLat} +lon_0={SensorLon} +ellps=WGS84 +datum=WGS84 +units=m +type=crs"
    )
    gps_to_sensor = Transformer.from_crs(wgs_84, sensor_position, always_xy=True)

    path = kwargs["filepath"]
    df = pd.read_csv(path)

    ground_truth = GroundTruthPath()

    for _, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]

        time = row["datetime(utc)"]
        timestamp = datetime.strptime(time, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=38)

        x, y = gps_to_sensor.transform(lon, lat)

        # offset angle
        angle_offset = np.deg2rad(70)
        x_offset = 2.6
        y_offset = 5.4
        x_final = x * np.cos(angle_offset) - y * np.sin(angle_offset) + x_offset
        y_final = x * np.sin(angle_offset) + y * np.cos(angle_offset) + y_offset

        truth_state = (
            State(
                state_vector=StateVector([x_final, 0, y_final, 0]),
                timestamp=timestamp,
            )
            if kwargs["model"] == "cv"
            else State(
                state_vector=StateVector([x_final, 0, 0, y_final, 0, 0]),
                timestamp=timestamp,
            )
        )
        ground_truth.append(truth_state)

    return [ground_truth]


def interpolate_ground_truth(ground_truth, timestamps, model):
    original_values = np.array([state.state_vector.flatten() for state in ground_truth])
    original_times = np.array([state.timestamp.timestamp() for state in ground_truth])

    target_times = np.array([ts.timestamp() for ts in timestamps])

    num_dims = 4 if model == "cv" else 6
    interpolated_values = np.zeros((len(target_times), num_dims))

    pos_dims = [0, 2] if model == "cv" else [0, 3]

    for dim in pos_dims:
        interpolated_values[:, dim] = np.interp(
            target_times, original_times, original_values[:, dim]
        )

    interpolated_path = GroundTruthPath()
    for ts, state_vec in zip(timestamps, interpolated_values):
        new_state = GroundTruthState(
            state_vector=state_vec.reshape(-1, 1), timestamp=ts
        )
        interpolated_path.append(new_state)

    return [interpolated_path]

def gps_to_cartesian(**kwargs):

    SensorLat = 51.98880548
    SensorLon = 4.390007015

    calibrationlat = 51.98837043001157

    calibrationlon = 4.3902004657738125
    y_cal = 49.97
    x_cal = 0

    wgs_84 = CRS("EPSG:4326")
    sensor_position = CRS(
        f"+proj=ortho +lat_0={SensorLat} +lon_0={SensorLon} +ellps=WGS84 +datum=WGS84 +units=m +type=crs"
    )
    gps_to_sensor = Transformer.from_crs(wgs_84, sensor_position)


    path = kwargs['filepath']
    df = pd.read_csv(path)

    print(df.iloc[:,3].values)
    # x, y = gps_to_sensor.transform(df.iloc[1, 3], df.iloc[1, 2])

    x, y = gps_to_sensor.transform(df.iloc[:,2], df.iloc[:,3])


    pointx, pointy = gps_to_sensor.transform(calibrationlat, calibrationlon)

    print(pointx, pointy)
    # print(df.iloc[:,0])
    # print(df.iloc[:, 2])
    # print(df.iloc[:, 3])


    center_angle = np.arctan2(pointy, pointx)
    fov_angle_1 = center_angle + np.radians(45)
    fov_angle_2 = center_angle - np.radians(45)
    R = 20

    line1_x = [0, R * np.cos(fov_angle_1)]
    line1_y = [0, R * np.sin(fov_angle_1)]
    line2_x = [0, R * np.cos(fov_angle_2)]
    line2_y = [0, R * np.sin(fov_angle_2)]

    plt.scatter(0, 0, color='red', s=50, label='Sensor')
    plt.plot(line1_x, line1_y, 'g--', label='FOV Border (+45°)')
    plt.plot(line2_x, line2_y, 'g--', label='FOV Border (-45°)')
    print(np.sqrt(pointx ** 2 + pointy ** 2))
    plt.scatter(x, y)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.title("Cartesian plot of the GPS track")

    return x, y

if __name__ == "__main__":
    gps_to_ground_truth(
        filepath="../Data/GPSdata/May-22nd-2026-10-27AM-Flight-Airdata.csv", model="ca"
    )
