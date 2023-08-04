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


# Simulation
def simulation_data():
    res = Results(
        code = "1647571402",
        group = "dreon",
        data_dir = "/home/nicolau/MSc/results/dy-dreon/data/simulation/")

    cm = res.centre_of_mass(axis = [2])
    x = cm[0] * 136e-3
    y = cm[1] * 10

    return (
        x,
        y,
        "Simulation")
