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
from data import centre_of_mass_data, centre_of_mass_experiment, centre_of_mass_theory

centre_of_mass_theory(gamma = 7.5e3, s = 248),

single_plot(
    centre_of_mass_data(),
    centre_of_mass_experiment(gamma = 7.5e3, s = 248),
    centre_of_mass_theory(gamma = 2*np.pi*7.5e3, s = 248),
    x_label=r"$\Delta\ [\Gamma']$",
    y_label=r"$z_c\ [mm]$",
    figsize = (6,4.5),
    fontsize = 16)

