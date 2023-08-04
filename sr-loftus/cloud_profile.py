# Libraries
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('auxiliar/')
sys.path.append(os.path.abspath(os.path.pardir) + "/auxiliar/")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from plots import heatmap_subplots
from data import cloud_profile_data

data = []
for idx in [14, 8, 2]:
    grid, bins, detuning = cloud_profile_data(idx)
    data.append({"grid": grid, "bins": bins, "detuning": detuning})

heatmap_subplots(
    *data,
    top_cut = 0.2,
    bottom_cut = 0.2,
    side_cut = 0.2,
    fontsize = 14)
