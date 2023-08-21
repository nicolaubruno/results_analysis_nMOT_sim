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
    data_dir = "/media/nicolau/Data/USP/Postgraduate/MSc/MOT Simulation/results/repository/sr-loftus/data/simulation/")

# Centre of mass
def centre_of_mass_data():
    cm = res.centre_of_mass(axis = [2])
    x = cm[0] / np.sqrt(1 + res.beams["main"]["s_0"])
    y = cm[1] * 10

    return (
        x,
        y,
        "Simulation")

# Experimental centre of mass
def centre_of_mass_experiment(gamma, s):
    df = pd.read_csv("/media/nicolau/Data/USP/Postgraduate/MSc/MOT Simulation/results/repository/sr-loftus/data/experiment/Vertical Position vs Laser Detuning/paper_data.csv")
    X = -df.iloc[:,0] * 1e6 / (gamma * np.sqrt(1 + s)) # power-broadened linewidth
    Y = df.iloc[:,1] # millimetre

    return (
        X,
        Y,
        "Experiment")

# Theoretical centre of mass
def centre_of_mass_theory(gamma, s):
    df = pd.read_csv("/media/nicolau/Data/USP/Postgraduate/MSc/MOT Simulation/results/repository/sr-loftus/data/experiment/Vertical Position vs Laser Detuning/paper_data.csv")
    X = - 2 * np.pi * df.iloc[:,0] * 1e6 / (gamma * np.sqrt(1 + s)) # power-broadened linewidth

    R = 15.806916768162177
    muB = 9.2740100783e-24
    hbar = 1.054571817e-34
    beta = 10e-2 * (muB / hbar) * 1.5
    x_int = np.linspace(np.min(X), np.max(X), 1000)
    Y = ((X*(gamma * np.sqrt(1 + s)) + (gamma / 2) * np.sqrt(R * s - 1 - s)) / beta)*1e3

    return (
        X,
        Y,
        "Theory",
        False)

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

# Centre of mass
def estimated_temperature():
    Y = res.temperature() * 1e6
    X = np.array(res.loop["values"], dtype="float") / np.sqrt(1 + res.beams["main"]["s_0"])

    return (
        X,
        Y,
        "Simulation")


# Experimental centre of mass
def experimental_temperature(gamma, s):
    df = pd.read_csv("/media/nicolau/Data/USP/Postgraduate/MSc/MOT Simulation/results/repository/sr-loftus/data/experiment/temperature_detuning.csv")
    X = - 2 * np.pi * df.iloc[:,0] * 1e6 / (gamma * np.sqrt(1 + s)) # power-broadened linewidth
    Y = df.iloc[:,1] # uK

    return (
        X,
        Y,
        "Experiment")
