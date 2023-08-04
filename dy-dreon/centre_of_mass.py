# Libraries
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('auxiliar/')
sys.path.append(os.path.abspath(os.path.pardir) + "/auxiliar/")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from plots import single_plot
from data import centre_of_mass_data

experiment_B_1_71_df = pd.read_csv(
    "../data/experiment/centre_of_mass_B_1_71.csv",
    header=0)

experiment_B_1_71 = (
    experiment_B_1_71_df.iloc[:,0].to_numpy(),
    experiment_B_1_71_df.iloc[:,1].to_numpy(),
    "Experiment")

single_plot(
    experiment_B_1_71,
    centre_of_mass_data(),
    x_label=r"$\Delta\ [2\pi \times MHz]$",
    y_label=r"$z_c\ [mm]$")
