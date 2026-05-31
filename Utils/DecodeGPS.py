import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#for the gps transformation, coordinate reference system en transformer -> reference is bij de radar (0,0)
from pyproj import CRS, Transformer


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

if __name__ == '__main__':
    gps_to_cartesian(filepath="../Data/GPSdata/May-22nd-2026-10-27AM-Flight-Airdata.csv")
