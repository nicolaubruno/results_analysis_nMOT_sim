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
from data import *

single_plot(
    estimated_temperature(),
    experimental_temperature(2*np.pi*7.5e3, 248),
    x_label=r"$\Delta\ [\Gamma']$",
    y_label=r"$T\ [\mu K]$",
    figsize = (6,4.5))

