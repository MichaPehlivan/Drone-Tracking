import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_track(trueTrack, dt):

    fig, ax = plt.subplots(figsize=(8, 8))


    ax.set_xlim(np.min(trueTrack[0, :]) - 5, np.max(trueTrack[0, :]) + 5)
    ax.set_ylim(np.min(trueTrack[1, :]) - 5, np.max(trueTrack[1, :]) + 5)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("True Track Animation")

    line, = ax.plot([], [], 'g--', alpha=0.5, label="Path History")
    point, = ax.plot([], [], 'go', markersize=8, label="Object")
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.legend()

    def init():
        line.set_data([], [])
        point.set_data([], [])
        time_text.set_text('')
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


    ani = FuncAnimation(fig, update, frames=trueTrack.shape[1],
                        init_func=init, blit=True, interval=dt * 1000)

    plt.show()
    return ani