# Libraries
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('auxiliar/')
sys.path.append(os.path.abspath(os.path.pardir) + "/auxiliar/")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from Results import Results

res = Results(
    code = "1647577959",
    group = "loftus",
    data_dir = "/home/nicolau/MSc/results/sr-loftus/data/simulation/")

# Centre of mass
def centre_of_mass_data(gamma = 136e-3):
    cm = res.centre_of_mass(axis = [2])
    x = cm[0] * sqrt(1 + res.beams["main"]["s_0"])
    y = cm[1] * 10

    return (
        x,
        y,
        "Simulation")

# Cloud size
def cloud_size_data(axis = 2, label = r"$\sigma$"):
    cm = res.centre_of_mass(axis = [axis])
    x = cm[0] / np.sqrt(1 + res.beams["main"]["s_0"])
    std = cm[2] * 100

    return (
        x,
        std,
        label)

# Cloud profile
def cloud_profile_data(idx):
    res.loop_idx(idx)
    pos = res.pos_hist
    detuning = res.beams["main"]["delta"] / np.sqrt(1 + res.beams["main"]["s_0"])

    x = pos[0]["dens"]
    z = pos[2]["dens"]
    size = x.size
    bins = pos[0]["bins"]
    grid = np.zeros((size, size))

    for i in range(x.size):
        for j in range(z.size):
            grid[j][i] = x[i]*z[size - j - 1]

    return grid, bins, detuning


