# External
from matplotlib import pyplot as plt


def plotSimpleKalman(x_history, measurements, trueTrack, ospa_distance):

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 2,
            "axes.linewidth": 1.2,
            "figure.dpi": 100,
            "savefig.dpi": 300,
        }
    )

    textstr = f"Average OSPA: {ospa_distance:.3f}"

    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black")

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.text(
        0.95,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    ax.scatter(
        measurements[0, :],
        measurements[1, :],
        color="red",
        marker="x",
        s=80,
        linewidths=1.5,
        label="Measurements",
        zorder=3,
    )
    ax.plot(
        x_history[0, :],
        x_history[1, :],
        color="blue",
        label="Kalman Filter Track",
        zorder=4,
    )
    ax.plot(
        trueTrack[0, :],
        trueTrack[1, :],
        color="green",
        label="True track",
        linestyle="--",
        zorder=5,
    )

    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper left", frameon=True, edgecolor="black")
    ax.minorticks_on()

    ax.set_title("Kalman Filter Track")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    plt.tight_layout()
    plt.show()

    return


def plotJointKalman(
    x_history_ucmkf,
    x_history_ekf,
    x_history_ukf,
    measurements,
    trueTrack,
    ospa_distance_ucmkf,
    ospa_distance_ekf,
    ospa_distance_ukf,
):

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 2,
            "axes.linewidth": 1.2,
            "figure.dpi": 100,
            "savefig.dpi": 300,
        }
    )

    textstr_ucmkf = f"Average OSPA UCMKF: {ospa_distance_ucmkf:.3f}"
    textstr_ekf = f"Average OSPA EKF: {ospa_distance_ekf:.3f}"
    textstr_ukf = f"Average OSPA UKF: {ospa_distance_ukf:.3f}"

    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black")

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.text(
        0.95,
        0.05,
        textstr_ukf,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )
    ax.text(
        0.95,
        0.1,
        textstr_ekf,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )
    ax.text(
        0.95,
        0.15,
        textstr_ucmkf,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    ax.scatter(
        measurements[0, :],
        measurements[1, :],
        color="red",
        marker="x",
        s=80,
        linewidths=1.5,
        label="Measurements",
        zorder=3,
    )
    ax.plot(
        x_history_ucmkf[0, :],
        x_history_ucmkf[1, :],
        color="red",
        label="UCMKF Track",
        zorder=4,
    )
    ax.plot(
        x_history_ekf[0, :],
        x_history_ekf[1, :],
        color="blue",
        label="EKF Track",
        zorder=4,
    )
    ax.plot(
        x_history_ukf[0, :],
        x_history_ukf[1, :],
        color="purple",
        label="UKF Track",
        zorder=4,
    )
    ax.plot(
        trueTrack[0, :],
        trueTrack[1, :],
        color="green",
        label="True track",
        linestyle="--",
        zorder=5,
    )

    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper left", frameon=True, edgecolor="black")
    ax.minorticks_on()

    ax.set_title(
        "Tracking Performance: Truth vs. UCMKF vs. EKF vs. UKF vs. Measurements"
    )
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    plt.tight_layout()
    plt.show()

    return
