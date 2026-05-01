# External
from matplotlib import pyplot as plt




def plotSimpleKalman(x_history, measurements):

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 11,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'axes.linewidth': 1.2,
        'figure.dpi': 100,
        'savefig.dpi': 300
    })

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(measurements[0, :], measurements[1, :], color='red', marker='x', s=80, linewidths=1.5, label='Measurements', zorder=3)
    ax.plot(x_history[0, :], x_history[1, :],color='blue', label='Kalman Filter Track', zorder=4)

    ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.8)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper left', frameon=True, edgecolor='black')
    ax.minorticks_on()

    ax.set_title('Kalman Filter Track')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')

    plt.tight_layout()
    plt.show()

    return