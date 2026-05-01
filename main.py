import numpy as np
import matplotlib.pyplot as plt

drone_x = 0
drone_y = 0
drone_v_x = 5
drone_v_y = 2

N = 20

A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
Q = np.eye(4) * 0
R = np.eye(2) * 1
P = np.eye(4) * 100

x = np.array([[0, 0, 0, 0]]).T
track = []
measurements = []
pos_estimates = []
vel_estimates = []
for _ in range(N):
    drone_x += drone_v_x
    drone_y += drone_v_y
    track.append((drone_x, drone_y))

    # Add random Gaussian noise to the measurement
    z = np.array([[drone_x], [drone_y]]) + np.random.normal(0, 1, (2, 1))
    measurements.append(z.flatten())

    # predict
    x = A @ x
    P = A @ P @ A.T + Q

    # update
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x = x + K @ (z - H @ x)
    P = (np.eye(4) - K @ H) @ P

    pos_estimates.append((x[0, 0], x[1, 0]))
    vel_estimates.append((x[2, 0], x[3, 0]))

true_track = np.array(track)
measurements = np.array(measurements)
kalman_track = np.array(pos_estimates)

plt.plot(
    true_track[:, 0], true_track[:, 1], "g-", label="True Track", alpha=0.3, linewidth=3
)
plt.scatter(
    measurements[:, 0],
    measurements[:, 1],
    marker="x",
    color="red",
    label="Measurements",
)
plt.plot(kalman_track[:, 0], kalman_track[:, 1], "b-", label="Kalman Filter Track")

# Formatting to match your image
plt.title("Kalman Filter Tracking")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.axis("equal")  # Keeps the scale 1:1
plt.show()

# 3. Plotting Velocity
vel_est_np = np.array(vel_estimates)
steps = np.arange(N)

plt.figure(figsize=(10, 5))

# Plot Vx
plt.plot(steps, vel_est_np[:, 0], "b-", label="Estimated $V_x$")
plt.axhline(y=drone_v_x, color="b", linestyle="--", alpha=0.5, label="True $V_x$")

# Plot Vy
plt.plot(steps, vel_est_np[:, 1], "r-", label="Estimated $V_y$")
plt.axhline(y=drone_v_y, color="r", linestyle="--", alpha=0.5, label="True $V_y$")

plt.title("Kalman Filter Velocity Convergence")
plt.xlabel("Time Step")
plt.ylabel("Velocity (units/step)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
