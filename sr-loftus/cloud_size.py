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
from data import cloud_size_data

single_plot(
    cloud_size_data(0, label = r"$\sigma_{x}$"),
    cloud_size_data(1, label = r"$\sigma_{y}$"),
    cloud_size_data(2, label = r"$\sigma_{z}$"),
    x_label=r"$\Delta\ [\Gamma']$",
    y_label=r"$\sigma\ [nm]$",
    figsize = (5,4))
