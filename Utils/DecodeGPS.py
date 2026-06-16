from datetime import datetime, timedelta
from matplotlib.animation import PillowWriter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
# for the gps transformation, coordinate reference system en transformer -> reference is bij de radar (0,0)
from pyproj import CRS, Transformer

from stonesoup.types.state import State
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState


def gps_to_ground_truth(**kwargs):
    delay = kwargs["gps_offset_delay"]
    SensorLat = 51.98880548
    SensorLon = 4.390007015
    start_time = kwargs["start_time"] - timedelta(seconds=delay)
    i=0
    wgs_84 = CRS("EPSG:4326")
    sensor_position = CRS(
        f"+proj=ortho +lat_0={SensorLat} +lon_0={SensorLon} +ellps=WGS84 +datum=WGS84 +units=m +type=crs"
    )
    gps_to_sensor = Transformer.from_crs(wgs_84, sensor_position, always_xy=True)

    path = kwargs["filepath"]
    df = pd.read_csv(path)

    delay = kwargs["gps_offset_delay"]

    timelength = timedelta(milliseconds=int(df.iloc[-1, 0] - df.iloc[0,0]))

    ground_truth = GroundTruthPath()
    ts = []
    for _, row in df.iterrows():

        lat = row["latitude"]
        lon = row["longitude"]

        # time = row["datetime(utc)"]
        # timestamp = datetime.strptime(time, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=38)
        timestamp = start_time + i*timedelta(milliseconds=100) # 100 ms per GPS thing
        i+=1
        x, y = gps_to_sensor.transform(lon, lat)

        # offset angle
        angle_offset = np.deg2rad(72.6)
        x_offset = 0
        y_offset = 0
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
        ts.append(timestamp)
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


    center_angle = np.arctan2(pointy, pointx)
    fov_angle_1 = np.radians(45)
    fov_angle_2 = -np.radians(45)
    R = 20
    x_rotated = x * np.cos(-center_angle) - y * np.sin(-center_angle)
    y_rotated = x * np.sin(-center_angle) + y * np.cos(-center_angle)

    line1_x = [0, R * np.cos(fov_angle_1)]
    line1_y = [0, R * np.sin(fov_angle_1)]
    line2_x = [0, R * np.cos(fov_angle_2)]
    line2_y = [0, R * np.sin(fov_angle_2)]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_title("Cartesian plot of the GPS track")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)


    ax.scatter(0, 0, color='red', s=50, label='Sensor', zorder=5)
    ax.plot(line1_x, line1_y, 'g--', label='FOV Border (45°)')
    ax.plot(line2_x, line2_y, 'g--')

    track_scatter = ax.scatter([], [], color='blue', alpha=0.6)

    ax.set_xlim(-1,15)
    ax.set_ylim(-8,8)
    ax.legend()

    def update(frame):

        data = np.stack((x_rotated[:frame + 1], y_rotated[:frame + 1]), axis=-1)
        track_scatter.set_offsets(data)

        current_x, current_y = x_rotated[frame], y_rotated[frame]
        distance = np.sqrt(current_x ** 2 + current_y ** 2)
        angle = np.arctan2(current_y, current_x)
        print(f"Frame {frame:03d} | Distance: {distance:.4f} | Angle: {angle:.4f}")

        return track_scatter,


    ani = FuncAnimation(
        fig,
        update,
        frames=len(x),
        interval=100,
        blit=True,
        repeat=True
    )

    plt.show()
    # xlist = []
    # ylist = []
    #
    # fig = plt.figure()
    # l, = plt.plot([],[], "k-")

    # writer = PillowWriter(fps=10)
    # with writer.saving(fig, "gpstrack.gif", 100):
    #     for xval in x:
    #         xlist.append(xval)
    #         ylist.append(1)
    #
    #         l.set_data(xlist,ylist)
    #         writer.grab_frame()

    # plt.scatter(0, 0, color='red', s=50, label='Sensor')
    # plt.plot(line1_x, line1_y, 'g--', label='FOV Border (+45°)')
    # plt.plot(line2_x, line2_y, 'g--', label='FOV Border (-45°)')
    # print(np.sqrt(pointx ** 2 + pointy ** 2))
    # plt.scatter(x, y)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    #
    # plt.title("Cartesian plot of the GPS track")

    return x, y

if __name__ == "__main__":


    # gps_to_cartesian(filepath="../Data/Hovering/flight6-hovering-GPS.csv")
    gps_to_cartesian(filepath="../Data/GPSdata/May-22nd-2026-10-27AM-Flight-Airdata.csv")