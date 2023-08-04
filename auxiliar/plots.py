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

    # Settings
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({"font.size":14})

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)

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

    plt.legend(frameon=True)
    plt.grid(linestyle="--")

    # Style
    ax = plt.gca()
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(2)
    ax.tick_params(labelsize=14)
    plt.tight_layout()

    plt.show()

def plot_heatmap(data, bins, x_label = "x [mm]", y_label = "z [mm]", top_cut = 0.0, bottom_cut = 0.0, side_cut = 0.0):
    # Clear plot
    plt.clf()

    # Settings
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({"font.size":14})
    #plt.legend(frameon=True)
    plt.grid(linestyle="")

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    size = bins.size
    data = data[int(size*top_cut):int((1 - bottom_cut)*size - 1), int(size*side_cut):int((1 - side_cut)*size - 1)]
    #ybins = bins[int(size*top_cut):]
    xbins = bins[int(size*side_cut):int((1 - side_cut)*size - 1)]
    ybins = np.flip(bins)[int(size*top_cut):int((1 - bottom_cut)*size - 1)]

    # Plot data
    plt.imshow(
        np.array(data)/np.max(data),
        cmap = "binary",
        extent = [min(xbins)*10, max(xbins)*10, min(ybins)*10, max(ybins)*10])

    # Style
    ax = plt.gca()
    #ax.spines[["top", "right"]].set_visible(False)
    #ax.spines[["bottom", "left"]].set_linewidth(2)
    ax.tick_params(labelsize=14)
    plt.tight_layout()

    plt.show()
    plt.close()


def heatmap_subplots(*args, **kwargs):
    # Clear plot
    plt.clf()

    # Arguments
    x_label = kwargs["x_label"] if "x_label" in kwargs else "x [mm]"
    y_label = kwargs["y_label"] if "y_label" in kwargs else "z [mm]"
    top_cut = kwargs["top_cut"] if "top_cut" in kwargs else 0
    bottom_cut = kwargs["bottom_cut"] if "bottom_cut" in kwargs else 0
    side_cut = kwargs["side_cut"] if "side_cut" in kwargs else 0
    figsize = kwargs["figsize"] if "figsize" in kwargs else (7,6)
    fontsize = kwargs["fontsize"] if "fontsize" in kwargs else 14

    # Settings
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({"font.size":fontsize})
    fig, axs = plt.subplots(1, len(args), sharey = True, figsize = figsize)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for i in range(len(args)):
        detuning = args[i]["detuning"]
        bins = args[i]["bins"]
        size = bins.size
        grid = args[i]["grid"][int(size*top_cut):int((1 - bottom_cut)*size - 1), int(size*side_cut):int((1 - side_cut)*size - 1)]
        xbins = bins[int(size*side_cut):int((1 - side_cut)*size - 1)]*10 #mm
        ybins = np.flip(bins)[int(size*top_cut):int((1 - bottom_cut)*size - 1)]*10 #mm

        axs[i].imshow(
            np.array(grid)/np.max(grid),
            cmap = "viridis",
            extent = [min(xbins), max(xbins), min(ybins), max(ybins)])

        axs[i].set_xlabel(x_label, fontsize=fontsize)
        if i == 0: axs[i].set_ylabel(y_label, fontsize=fontsize)
        axs[i].grid(linestyle="")
        axs[i].text(
            np.min(xbins)*0.8, np.max(ybins)*0.7,
            r"${} {:.1f}\ \Gamma' $".format(labels[i], detuning),
            color = "white",
            fontsize = fontsize)

    # Style
    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()

    plt.close(1)
    plt.show()
