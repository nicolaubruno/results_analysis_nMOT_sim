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

single_plot(
    centre_of_mass_data(),
    x_label=r"$\Delta\ [2\pi \times MHz]$",
    y_label=r"$z_c\ [mm]$")
