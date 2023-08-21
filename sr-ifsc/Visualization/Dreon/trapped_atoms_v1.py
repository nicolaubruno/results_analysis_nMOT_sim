# Libraries
import numpy as np
from matplotlib import pyplot as plt, rc
import pandas as pd
import seaborn as sns

# Data
#--
# Experiment EOM on
#--
exp_on_data = pd.read_csv("../../Dreon/Experiment/atoms_number_vs_detuning_EOM_on.csv").astype(float)
exp_on_data.index = np.arange(1, len(exp_on_data) + 1, 1)

exp_on_X = np.array(exp_on_data.iloc[:,0].tolist())
exp_on_X = exp_on_X*1e3 / 136 # Detuning [Gamma]

exp_on_Y = np.array(exp_on_data.iloc[:,1].tolist())
exp_on_Y = exp_on_Y / np.max(exp_on_Y) # N / N_{max}
#--

# Experiment EOM off
#--
exp_off_data = pd.read_csv("../../Dreon/Experiment/atoms_number_vs_detuning_EOM_off.csv").astype(float)
exp_off_data.index = np.arange(1, len(exp_off_data) + 1, 1)

exp_off_X = np.array(exp_off_data.iloc[:,0].tolist())
exp_off_X = exp_off_X*1e3 / 136 # Detuning [Gamma]

exp_off_Y = np.array(exp_off_data.iloc[:,1].tolist())
exp_off_Y = exp_off_Y / np.max(exp_off_Y) # N / N_{max}
#--

# Simulation EOM off
#--
sim_off_data = pd.read_csv("../../Dreon/Simulation/CSV Files/trap_depth_EOM_off_0.5.csv").astype(float)
sim_off_data.index = np.arange(1, len(sim_off_data) + 1, 1)

sim_off_X = np.array(sim_off_data.iloc[:,0].tolist()) # Detuning [Gamma]

sim_off_Y = np.array(sim_off_data.iloc[:,1].tolist()) # Temperature [mK]
sim_off_Y = sim_off_Y / np.max(sim_off_Y) # T / T_max
#--

# Simulation EOM on
#--
sim_on_data = pd.read_csv("../../Dreon/Simulation/CSV Files/trap_depth_EOM_on_0.5.csv").astype(float)
sim_on_data.index = np.arange(1, len(sim_on_data) + 1, 1)

sim_on_X = np.array(sim_on_data.iloc[:,0].tolist()) # Detuning [Gamma]

sim_on_Y = np.array(sim_on_data.iloc[:,1].tolist()) # Temperature [mK]
sim_on_Y = sim_on_Y / np.max(sim_on_Y)
#--
#--

# Theoretical Cut
#--
R = 169.28111202 # Maximum radiation pressure force divided by gravity
s = 50

B_0 = 1.7e-4 # T / cm
mu_B = 927.4009994e-26 # J / T
hbar = 1.054571800e-34 # J s

beta = (mu_B / hbar) * (B_0 / 2)
chi = 1.29*9 - 1.24*8
w = 2.0 # cm

max_delta_cut = -np.sqrt((R - 1)*s - 1) / 2
min_delta_cut = max_delta_cut - (beta * chi * w) / (2 * np.pi * 136e3)
#--

# Plotting
#--
# Set plot
plt.clf()
plt.figure(figsize=(11,5))
plt.style.use("seaborn-whitegrid")
rc("font", size=12)
rc("axes", labelsize=14)
rc("xtick", labelsize=14)
rc("ytick", labelsize=14)

# EOM Off
#--
plt.subplot(1, 2, 1)

# Set labels
plt.title("EOM Off")
plt.xlabel(r"$\Delta [\Gamma]$")
plt.ylabel(r"$ T_c [mK] $")

# Simulation
#--
# Max Point of the Simulation
max_X_idx = np.where(sim_off_Y == np.max(sim_off_Y))[0][0] 
plt.plot(sim_off_X, sim_off_Y, color=sns.color_palette()[0], linestyle="", marker="o", label=(("Sim. (Cut off: " + r"$%.1f \Gamma$" + ")") % sim_off_X[max_X_idx]))
#--

# Experiment
#--
# Max Point of the Experiment
max_X_idx = np.where(exp_off_Y == np.max(exp_off_Y))[0][0]
plt.plot(exp_off_X, exp_off_Y, color=sns.color_palette()[3], linestyle="", marker="s", label=(("Exp. (Cut off: " + r"$%.1f \Gamma$" + ")") % exp_off_X[max_X_idx]))
#--

# Cut off laser detuning estimative
plt.fill_between([min_delta_cut, max_delta_cut], 0-max(sim_off_Y)*0.1, max(sim_off_Y)*1.1, color=sns.color_palette()[7], alpha=0.4, label="Cut off estimative")

# Set visualization
plt.ylim(0-max(sim_off_Y)*0.1, max(sim_off_Y)*1.1)
plt.xlim(min_delta_cut, max(exp_off_X)*0.8)
plt.legend(frameon=True, loc="lower left")
plt.grid(linestyle="--")
#--

# Plot (1, 2, 2)
#--
plt.subplot(1, 2, 2)

# Simulation
#--
# Max Point of the Simulation
max_X_idx = np.where(sim_on_Y == np.max(sim_on_Y))[0][0] 
plt.plot(sim_on_X, sim_on_Y, color=sns.color_palette()[0], linestyle="", marker="o", label=(("Sim. (Cut off: " + r"$%.1f \Gamma$" + ")") % sim_on_X[max_X_idx]))
#--

# Experiment
#--
# Max Point of the Experiment
max_X_idx = np.where(exp_on_Y == np.max(exp_on_Y))[0][0]
plt.plot(exp_on_X, exp_on_Y, color=sns.color_palette()[3], linestyle="", marker="s", label=(("Exp. (Cut off: " + r"$%.1f \Gamma$" + ")") % exp_on_X[max_X_idx]))
#--

# Set labels
plt.title("EOM On")
plt.xlabel(r"$\Delta [\Gamma]$")
plt.ylabel(r"$ N / N_{max} $")

# Set visualization
plt.legend(frameon=True, loc="lower left")
plt.grid(linestyle="--")
#--

# Show
plt.close(1)
plt.tight_layout()
plt.show()
#--