import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_track(trueTrack, dt):

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(np.min(trueTrack[0, :]) - 5, np.max(trueTrack[0, :]) + 5)
    ax.set_ylim(np.min(trueTrack[1, :]) - 5, np.max(trueTrack[1, :]) + 5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("True Track Animation")

    (line,) = ax.plot([], [], "g--", alpha=0.5, label="Path History")
    (point,) = ax.plot([], [], "go", markersize=8, label="Object")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    ax.legend()

    def init():
        line.set_data([], [])
        point.set_data([], [])
        time_text.set_text("")
        return line, point, time_text

    def update(frame):
        # Extract x and y up to the current frame
        x = trueTrack[0, :frame]
        y = trueTrack[1, :frame]

        line.set_data(x, y)

        if frame > 0:
            point.set_data([trueTrack[0, frame - 1]], [trueTrack[1, frame - 1]])

        time_text.set_text(f"Time: {frame * dt:.1f}s")
        return line, point, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=trueTrack.shape[1],
        init_func=init,
        blit=True,
        interval=dt * 1000,
    )

    plt.show()
    return ani


def animate_TrackKalmanMeasurements(trueTrack, measurements, x_history, OSPA, dt):
    # Function expects everything in cartesian coordinates.
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlim(np.min(trueTrack[0, :]) - 5, np.max(trueTrack[0, :]) + 5)
    ax.set_ylim(np.min(trueTrack[1, :]) - 5, np.max(trueTrack[1, :]) + 5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Tracking Performance: Truth vs. EKF vs. Measurements")

    (line_true,) = ax.plot([], [], "g--", alpha=0.4, label="True Path")
    (point_true,) = ax.plot([], [], "go", markersize=6, label="True Object")

    (line_kalman,) = ax.plot([], [], "b-", linewidth=2, label="EKF Track")

    (meas_point,) = ax.plot(
        [], [], "rx", markersize=10, markeredgewidth=2, label="Measurement"
    )

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, weight="bold")

    ax.text(
        0.98,
        0.05,
        f"Average OSPA: {OSPA:.3f}",
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=12,
    )

    ax.legend(loc="upper right")

    def init():
        line_true.set_data([], [])
        point_true.set_data([], [])
        line_kalman.set_data([], [])
        meas_point.set_data([], [])
        time_text.set_text("")
        return line_true, point_true, line_kalman, meas_point, time_text

    def update(frame):

        x_true = trueTrack[0, :frame]
        y_true = trueTrack[1, :frame]
        line_true.set_data(x_true, y_true)

        if frame > 0:
            point_true.set_data([trueTrack[0, frame - 1]], [trueTrack[1, frame - 1]])

        x_kf = x_history[0, :frame]
        y_kf = x_history[1, :frame]
        line_kalman.set_data(x_kf, y_kf)

        if frame > 0:
            meas_point.set_data(
                [measurements[0, max(0, frame - 5) : frame]],
                [measurements[1, max(0, frame - 5) : frame]],
            )

        time_text.set_text(f"Time: {frame * dt:.1f}s")

        return line_true, point_true, line_kalman, meas_point, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=trueTrack.shape[1],
        init_func=init,
        blit=True,
        interval=dt * 1000,
    )

    plt.show()
    return ani
