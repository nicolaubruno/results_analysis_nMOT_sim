#
# Libraries
#--
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
import seaborn as sns

from Results import Results
#--

#
# Constants
#--
kB = 1.38064852e-23 # J / K
h = 6.626070040e-34 # J s
gamma = 7.6e3 # Hz / 2pi
s_regime_II = ((500/7.6)**2 - 1)/6
#--

#
# Data
#--
# Experiment (saturation)
data = pd.read_csv('../../data/strontium/power_scan.csv')[['s','T (uK)']];
s = data['s'].values
T = data['T (uK)'].values # uK
data_exp_1 = [s, T]

# Experiment (detuning)
data = pd.read_csv('../../data/strontium/frequency_scan.csv')[['detuning (MHz)','Temperature (uK)']];
delta = (data['detuning (MHz)'].values - 2.995) # 2pi x MHz
T = data['Temperature (uK)'].values # uK
data_exp_2 = [delta, T]

# Simulation (saturation)
#res = Results(1622255378, results_group="strontium") # Partial
res = Results(1622330709, results_group="strontium") # Complete
s = np.array(res.loop["values"])
T = res.temperature(method=0)*1e6 #uK
data_sim_1 = [s, T]

# Simulation (detuning)
res = Results(1622296705, results_group="strontium")
delta = np.array(res.loop["values"])*gamma*1e-6 # 2pi x MHz
T = res.temperature(method=0)*1e6 #uK
data_sim_2 = [delta, T]

# Theory (saturation)
def temperature_regime_I(s):
    return (h * gamma * np.sqrt(1 + s) / (kB))*1e6 #uK

s = np.linspace(data_sim_1[0][0], data_sim_1[0][-1], 1000)
T = temperature_regime_I(s) #uK
data_theory_1 = [s, T]

# Theory (detuning)
def temperature_regime_II(s, delta, scale_factor=1.0):
    gamma_E = gamma * np.sqrt(1 + s) * 1e-6 # 2pi x MHz
    T_D = (h * gamma * np.sqrt(1 + s) / (2*kB))*1e6 #uK

    return scale_factor * T_D * (1 + 4*(delta / gamma_E)**2) / (4 * np.abs(delta) / gamma_E) #uK

delta = np.linspace(data_exp_2[0][-1], data_sim_2[0][-7], 1000) # 2pi x MHz
T = temperature_regime_II(6*102, delta, 1.0)
data_theory_2 = [delta, T]

# Theory (detuning) - scaled
delta = np.linspace(data_exp_2[0][-1], data_sim_2[0][-7], 1000) # 2pi x MHz
T = temperature_regime_II(6*102, delta, 0.9)
data_theory_3 = [delta, T]
#--

gamma_E = gamma * np.sqrt(1 + 6*102)*1e-6

#
# Plot
#--
# Plot settings
plt.clf()
plt.style.use('seaborn-whitegrid') # Theme
plt.rcParams.update({'font.size':12})
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(9, 4.5) # Figure size

# Labels
title = ['(a)', '(b)']
for i in range(2):
    ax[i].set_title(title[i], loc="center", fontsize=14, fontweight="bold", pad=10)
    ax[i].set_yscale('log')
    ax[i].set_ylabel(r'$T\ [\mu K]$')
    ax[i].set_xlabel(r'$s$')
    ax[i].yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax[i].yaxis.set_minor_formatter(ticker.ScalarFormatter())

# Plots saturation
ax[0].plot(data_sim_1[0], data_sim_1[1], marker='s', linestyle="", color=sns.color_palette()[3], label="Simulation")
ax[0].plot(data_exp_1[0], data_exp_1[1], marker='o', linestyle="", color="black", label="Experiment")
ax[0].plot(data_theory_1[0], data_theory_1[1], marker='', linestyle="--", linewidth=2, color=sns.color_palette()[0], label="Theory (Regime I)")
ax[0].fill_between(np.linspace(s_regime_II, data_exp_1[0][-1]+10, 1000), 0, np.max(data_exp_1[1])+1, color=sns.color_palette()[4], alpha=0.4, label="Regime II")
ax[0].plot(data_theory_1[0], temperature_regime_II(6*data_theory_1[1], 0.5, 1.0), marker='', linestyle="--", linewidth=2, color=sns.color_palette()[2], label="Theory (Regime II)")
ax[0].set_ylim([1, np.max(data_exp_1[1])+1])
ax[0].set_xlim([0, data_exp_1[0][-1]+10])

# Plots detuning
ax[1].set_yscale('linear')
ax[1].set_xlabel(r'$\ \Delta / \Gamma_E $')
ax[1].plot(data_sim_2[0] / gamma_E, data_sim_2[1], marker='s', linestyle="", color=sns.color_palette()[3], label="Simulation")
ax[1].plot(data_exp_2[0] / gamma_E, data_exp_2[1], marker='o', linestyle="", color="black", label="Experiment")
#ax[1].plot(-np.ones(1000), np.linspace(0, data_exp_1[1][-1] + 0.5, 1000), linewidth=2, marker='', linestyle="-", color=sns.color_palette()[0], label=r"$|\Delta| = \Gamma_E$")
ax[1].plot(data_exp_2[0] / gamma_E, temperature_regime_I(102)*np.ones(len(data_exp_2[0])), marker='', linestyle="--", linewidth=2, color=sns.color_palette()[0], label="Theory (Regime I)")
ax[1].plot(data_theory_2[0] / gamma_E, data_theory_2[1], marker='', linestyle="--", linewidth=2, color=sns.color_palette()[2], label="Theory (Regime II)")
#ax[1].plot(data_theory_3[0], data_theory_3[1], marker='', linestyle="--", linewidth=2, color=sns.color_palette()[3], label="Theory (Regime II) with scaled factor")
ax[1].fill_between(np.linspace(-1, 0, 1000), 0, np.max(data_exp_2[1])+1, color=sns.color_palette()[4], alpha=0.4, label="Regime II")
ax[1].set_ylim([0, np.max(data_exp_2[1])+1])
ax[1].set_xlim([(np.min(data_exp_2[0]) / gamma_E) - 0.1, 0])

# Plot settings
for i in range(2):
    ax[i].grid(linestyle="--", which="both")
    ax[i].legend(frameon=True)

plt.tight_layout()
plt.close(1)
plt.show()
#--