import numpy as np
from datetime import datetime, timedelta
import pandas as pd

#External
from stonesoup.types.detection import Detection
from stonesoup.types.array import StateVector
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from sklearn.cluster import DBSCAN

#Function that wraps the Abdullah and Andrej measurements.
def ReadDetections(**kwargs):
    #get path from the keyword arguments
    path = kwargs['filepath']
    measurement_model = kwargs['measurement_model']
    dt = kwargs['dt']
    start_time = kwargs['start_time']

    #Read file into dataframe.
    df = pd.read_csv(path)

    grouped = df.groupby('block')

    all_detections = []



    for block_num, block_data in df.groupby('block'):

        timestamp = start_time + timedelta(seconds= block_num * dt)

        block_detections = set()

        for _, row in block_data.iterrows():
            det = Detection(
                state_vector=StateVector([np.deg2rad(row['angle_deg']), row['range_m']]),
                timestamp=timestamp,
                measurement_model=measurement_model
            )
            block_detections.add(det)

        all_detections.append(block_detections)

    # print(all_detections)
    return all_detections


def ReadAndClusterDetections(**kwargs):
    path = kwargs['filepath']
    measurement_model = kwargs['measurement_model']
    dt = kwargs['dt']
    start_time = kwargs['start_time']

    df = pd.read_csv(path)
    all_detections = []

    df['true_time'] = start_time + pd.to_timedelta(df['block'] * dt, unit='s')

    time_window = "0.3s"
    df = df.set_index('true_time').sort_index()

    for window_timestamp, block_data in df.groupby(pd.Grouper(freq=time_window)):
        if block_data.empty:
            all_detections.append(set())
            continue

        angles_rad = np.deg2rad(block_data['angle_deg'].values)
        ranges_m = block_data['range_m'].values

        x = ranges_m * np.cos(angles_rad)
        y = ranges_m * np.sin(angles_rad)
        cartesian_points = np.column_stack((x, y))

        # 3. DBSCAN now sees all points inside this time window simultaneously
        # It will independently choose how to group them based on their distance
        db = DBSCAN(eps=15.0, min_samples=1).fit(cartesian_points)
        labels = db.labels_

        block_detections = set()
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue

            cluster_mask = (labels == label)
            cluster_cartesian = cartesian_points[cluster_mask]

            centroid_cartesian = np.mean(cluster_cartesian, axis=0)
            cx, cy = centroid_cartesian[0], centroid_cartesian[1]

            centroid_range = np.sqrt(cx ** 2 + cy ** 2)
            centroid_angle = np.arctan2(cy, cx)

            det = Detection(
                state_vector=StateVector([centroid_angle, centroid_range]),
                timestamp=window_timestamp,  # Use the synchronized window timestamp
                measurement_model=measurement_model
            )
            block_detections.add(det)

        all_detections.append(block_detections)

    return all_detections


if __name__ == '__main__':

    range_sigma = 1
    azimuth_sigma = np.deg2rad(3)
    var_r = range_sigma ** 2
    var_phi = azimuth_sigma ** 2

    R = 1 * np.array([[var_phi, 0], [0, var_r]])

    # Define measurement_model
    measurement_model = CartesianToBearingRange(
        ndim_state=6,
        mapping=(0, 3),
        noise_covar=R
    )

    detections = ReadAndClusterDetections(filepath="../Data/NEO1_range_angle_velocity_detections_ndoppler50.csv",
                                measurement_model=measurement_model,
                                dt = 0.0112,
                                start_time=datetime.now()
    )
    print(len(list(detections)))


