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
    trueTrack = np.zeros((2, num_datapoints))
    measurements = np.zeros((2, num_datapoints))

    for i in range(num_datapoints):
        trueTrack[0, i] = x + v_x * t
        trueTrack[1, i] = y + v_y * t

        measurements[0, i] = trueTrack[0, i] + sigma * randn()
        measurements[1, i] = trueTrack[1, i] + sigma * randn()

        t += dt

    return measurements, trueTrack


"""
simulateLinearTrackPolar simulates a linear path taken by a drone

input:
    v_x: initial velocity in x-direction. (m/s)
    v_y: initial velocity in y-direction. (m/s)
    x0: initial x-position. (m)
    y0: initial y-position. (m)
    num_datapoints: number of datapoints in the track.
    dt: timestep (s).
    sigma: standard deviations of the measurement noise.

output:
    measurements: [2 x num_datapoints] array 
                  (bearing,range coordinates as rows, samples as columns)

                  containing simulated measurements for a straight-line drone track.
"""

def simulateLinearTrackPolar(v_x, v_y, x0, y0, num_datapoints, dt, sigma_r, sigma_phi):
    x = x0
    y = y0
    t = 0
    trueTrack = np.zeros((2, num_datapoints))
    measurements = np.zeros((2, num_datapoints))

    for i in range(num_datapoints):
        trueTrack[0, i] = x + v_x * t
        trueTrack[1, i] = y + v_y * t

        measurements[0, i] = (
            np.arctan2(trueTrack[1, i], trueTrack[0, i]) + sigma_phi * randn()
        )  # azimuth
        measurements[1, i] = (
            np.sqrt(trueTrack[0, i] ** 2 + trueTrack[1, i] ** 2) + sigma_r * randn()
        )  # range

        t += dt

    return measurements, trueTrack

"""
simulateRandomAccelTrackPolar simulates a linear path taken by a drone

input:
    v_x: initial velocity in x-direction. (m/s)
    v_y: initial velocity in y-direction. (m/s)
    x0: initial x-position. (m)
    y0: initial y-position. (m)
    num_datapoints: number of datapoints in the track.
    dt: timestep (s).
    sigma_r/phi: standard deviations of the measurement noise.
    measure_velocity: standard:False, whether to output velocity info.
output:
    measurements: [2 x num_datapoints] array 
                  (bearing,range coordinates as rows, samples as columns)

                  containing simulated measurements for a straight-line drone track.
"""
def simulateRandomAccelTrackPolar(
    v_x,
    v_y,
    x0,
    y0,
    num_datapoints,
    dt,
    sigma_r,
    sigma_phi,
    sigma_vr=0,
    measure_velocity=False,
):
    curr_x, curr_y = x0, y0
    curr_vx, curr_vy = v_x, v_y
    trueTrack = np.zeros((2, num_datapoints))
    measurements = np.zeros((3 if measure_velocity else 2, num_datapoints))
    v_max = 12
    g = 9.81

    maneuvering = False
    ax, ay = 0, 0

    for i in range(num_datapoints):

        if not maneuvering:
            if np.random.rand() < 0.10:
                maneuvering = True

                accel_mag = np.random.uniform(0.1 * g, 0.7 * g)

                theta = np.random.uniform(0, 2 * np.pi)

                ax, ay = accel_mag * np.cos(theta), accel_mag * np.sin(theta)

        else:
            if np.random.rand() < 0.30:
                maneuvering = False
                ax, ay = 0, 0

        curr_vx += ax * dt
        curr_vy += ay * dt

        v_mag = np.sqrt(curr_vx**2 + curr_vy**2)
        if v_mag > v_max:
            normalization = v_max / v_mag
            curr_vx = curr_vx * normalization
            curr_vy = curr_vy * normalization

        curr_x += curr_vx * dt
        curr_y += curr_vy * dt

        trueTrack[:, i] = [curr_x, curr_y]

        r = np.sqrt(curr_x**2 + curr_y**2)
        phi = np.arctan2(curr_y, curr_x)

        if measure_velocity:
            v_r = (curr_x * curr_vx + curr_y * curr_vy) / r
            measurements[:, i] = [
                phi + sigma_phi * randn(),
                r + sigma_r * randn(),
                v_r + sigma_vr * randn(),
            ]
        else:
            measurements[:, i] = [phi + sigma_phi * randn(), r + sigma_r * randn()]

    return measurements, trueTrack

"""
simulateRandomAccelHoverTrackPolar simulates a linear path taken by a drone

input:
    v_x: initial velocity in x-direction. (m/s)
    v_y: initial velocity in y-direction. (m/s)
    x0: initial x-position. (m)
    y0: initial y-position. (m)
    num_datapoints: number of datapoints in the track.
    dt: timestep (s).
    sigma_r/phi: standard deviations of the measurement noise.
    measure_velocity: standard:False, whether to output velocity info.
output:
    measurements: [2 x num_datapoints] array 
                  (bearing,range coordinates as rows, samples as columns)

                  containing simulated measurements for a straight-line drone track.
"""
def simulateRandomAccelHoverTrackPolar(
    v_x,
    v_y,
    x0,
    y0,
    num_datapoints,
    dt,
    sigma_r,
    sigma_phi,
    sigma_vr=0,
    measure_velocity=False,
):

    curr_x, curr_y = x0, y0
    curr_vx, curr_vy = v_x, v_y

    trueTrack = np.zeros((2, num_datapoints))
    measurements = np.zeros((3 if measure_velocity else 2, num_datapoints))

    v_max = 12 # m/s
    state = "MOVING"
    hover_timer = 0

    g = 9.81
    brake_accel = 0.8 * g

    ax, ay = 0, 0

    for i in range(num_datapoints):

        if state == "MOVING":

            if np.random.rand() < 0.1 * dt and i > 5:
                state = "BRAKING"

            elif np.random.rand() < dt:
                theta = np.random.uniform(0, 2 * np.pi)
                maneuver_accel = np.random.uniform(0.5 * g, 1.5 * g)
                ax, ay = maneuver_accel * np.cos(theta), maneuver_accel * np.sin(theta)

        if state == "BRAKING":

            v_mag = np.sqrt(curr_vx**2 + curr_vy**2)
            if v_mag > 5 * dt:
                ax = -(curr_vx / v_mag) * brake_accel
                ay = -(curr_vy / v_mag) * brake_accel
            else:
                ax, ay = 0, 0
                curr_vx, curr_vy = 0, 0
                state = "HOVERING"
                hover_timer = np.random.randint(5, 15)

        elif state == "HOVERING":
            ax, ay = 0, 0
            curr_vx, curr_vy = 0, 0
            hover_timer -= 1

            if hover_timer <= 0:
                state = "MOVING"
                launch_theta = np.random.uniform(0, 2 * np.pi)
                maneuver_accel = np.random.uniform(0.5 * g, 1.5 * g)
                ax, ay = maneuver_accel * np.cos(launch_theta), maneuver_accel * np.sin(
                    launch_theta
                )

        curr_vx += ax * dt
        curr_vy += ay * dt

        v_mag = np.sqrt(curr_vx**2 + curr_vy**2)
        if v_mag > v_max:
            normalization = v_max / v_mag
            curr_vx = curr_vx * normalization
            curr_vy = curr_vy * normalization

        curr_x += curr_vx * dt
        curr_y += curr_vy * dt

        trueTrack[:, i] = [curr_x, curr_y]
        r = np.sqrt(curr_x**2 + curr_y**2)
        phi = np.arctan2(curr_y, curr_x)

        if measure_velocity:
            v_r = (curr_x * curr_vx + curr_y * curr_vy) / r
            measurements[:, i] = [
                phi + sigma_phi * randn(),
                r + sigma_r * randn(),
                v_r + sigma_vr * randn(),
            ]
        else:
            measurements[:, i] = [phi + sigma_phi * randn(), r + sigma_r * randn()]

    return measurements, trueTrack

if __name__ == "__main__":
    np.random.seed(4)
    from Utils import animate_track
    import matplotlib.pyplot as plt
    detections, ground_truth = simulateRandomAccelHoverTrackPolar(
        v_x=1,
        v_y=1,
        x0=5,
        y0=5,
        num_datapoints=150,
        dt=0.1,
        sigma_r= np.sqrt(0.383),
        sigma_phi = np.sqrt(0.000585),
        sigma_vr=0,
        measure_velocity=False,
    )
    plt.scatter(detections[1, :]*np.cos(detections[0, :]), detections[1,: ]*np.sin(detections[0, :]), label="Detections")
    plt.grid()

    plt.title("Detections based on ground truth")
    plt.show()

    animate_track(ground_truth, 0.1)