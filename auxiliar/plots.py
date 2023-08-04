# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def single_plot(*data, **kwargs):
    # Arguments
    x_label = kwargs["x_label"] if "x_label" in kwargs else None
    y_label = kwargs["y_label"] if "y_label" in kwargs else None

    # Clear plot
    plt.clf()

    # Plot data
    markers = ["o", "v", "s", "X", "D"]
    for i in range(len(data)):
        x, y, label = data[i]

        plt.plot(
            x,
            y,
            marker=markers[i],
            markersize=10,
            markeredgewidth=1,
            markeredgecolor="black",
            linestyle="",
            label=label)

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    # Settings
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({"font.size":14})
    plt.legend(frameon=True)
    plt.grid(linestyle="--")

    # Style
    ax = plt.gca()
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(2)
    ax.tick_params(labelsize=14)
    plt.tight_layout()

    plt.show()
