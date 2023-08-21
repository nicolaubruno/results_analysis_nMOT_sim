#
# Libraries
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from Results import Results

#
# Data
#--
# Paper data
#--
del_B = "1.71" # G / cm
p_delta = []
p_std_dev = []

# y-axis
p_delta.append(np.array([-2.50, -2.15, -1.80, -1.45, -1.09, -0.74])) # 2*pi MHz
p_std_dev.append(np.array([1.07, 1.02, 0.95, 0.80, 0.64, 0.43])) # mm

# z-axis
p_delta.append(np.array([-2.50, -2.15, -1.80, -1.45, -1.09, -0.74])) # 2*pi MHz
p_std_dev.append(np.array([0.63, 0.59, 0.56, 0.52, 0.44, 0.35])) # mm
#--

#
# Simulation data
#--
# Small detunings
#s_res = Results(1616642739)
#s_z_c, s_std_dev = s_res.mass_centre(axis=[2])

# Middle detunings
#m_res = Results(1616088824)
#m_z_c, m_std_dev = m_res.mass_centre(axis=[2])

# Large detunings
#l_res = Results(1616591691)
#l_z_c, l_std_dev = l_res.mass_centre(axis=[2])

#
# Unique range
#--
# G = 2.26 G / cm
#res = Results(1616693405)

# G = 1.71 G / cm
#res = Results(1616805096)
#res = Results(1616891450)
res = Results(1616902620)
#res = Results(1617610575)
r_c, std_dev_r = res.mass_centre(axis=[1, 2])
#--
#--

#
# Concatenation
#--
# Detunings
#s_delta = np.array(s_res.loop["values"])*(s_res.transition["gamma"]*1e-3)
#m_delta = np.array(m_res.loop["values"])*(m_res.transition["gamma"]*1e-3)
#l_delta = np.array(l_res.loop["values"])*(l_res.transition["gamma"]*1e-3)

# Small + Middle detuning
#delta = np.concatenate((s_delta, m_delta), axis=None)
#z_c = np.concatenate((s_z_c, m_z_c), axis=None)
#std_z_c = np.concatenate((s_std_z_c, m_std_z_c), axis=None)

# Small + Large detuning
#delta = np.concatenate((s_delta, l_delta), axis=None)
#z_c = np.concatenate((s_z_c, l_z_c), axis=None)
#std_z_c = np.concatenate((s_std_z_c, l_std_z_c), axis=None)

# Middle + Large detuning
#delta = np.concatenate((m_delta, l_delta), axis=None)
#z_c = np.concatenate((m_z_c, l_z_c), axis=None)
#std_z_c = np.concatenate((m_std_z_c, l_std_z_c), axis=None)

# All detunings
#delta = np.concatenate((s_delta, m_delta, l_delta), axis=None)
#z_c = np.concatenate((s_z_c, m_z_c, l_z_c), axis=None)
#std_z_c = np.concatenate((s_std_z_c, m_std_z_c, l_std_z_c), axis=None)

# Unique range
delta = np.array(res.loop["values"])*(res.transition["gamma"]*1e-3)
#--
#--

#
# Visualization
#--
# Clear stored plots
plt.clf()

#
# Set style
#--
plt.style.use('seaborn-whitegrid')
#plt.subplots_adjust(left=2.0, right=3.0, top=0.02, bottom=0.01)
plt.rcParams.update({
        "font.size":14,\
        "axes.titlepad":14
    })

# Set labels
label = ['y', 'z']
#plt.title("Standard deviation as a function of \nthe laser detuning " + r"($\nabla B = " + del_B + " G/cm$)")
#plt.title("Ratio of standard deviations ("+ r"$ \sigma_y / \sigma_z $" +") as a function \nof the laser detuning " + r"($\nabla B = " + del_B + " G/cm$)")
#--

#
# Set plot figure
fig = plt.figure(figsize=(6,5))
gs = fig.add_gridspec(1, 2, wspace=0)
ax = gs.subplots(sharey=True)
fig.subplots_adjust(top=0.75)
fig.suptitle("Standard deviation as a function of \nthe laser detuning " + r"($\nabla B = " + del_B + " G/cm$)", size=16, y=0.95)
#fig.suptitle("Ratio of standard deviations of \nthe laser detuning " + r"($\nabla B = " + del_B + " G/cm$)", size=16, y=0.95)
plt.close(1)

#
# Plot
#--
for i in range(2):
    # Simulated points
    ax[i].plot(delta, std_dev_r[i]*10, label="Simulation", marker='o', linestyle='--')

    # Experiment
    ax[i].plot(p_delta[i], p_std_dev[i], label="Experiment", marker='^', linestyle='--')

    # Settings
    ax[i].set_title(r"$ \sigma_" + label[i] + " $")
    ax[i].set_xlabel(r"$ \Delta\ [2\pi \times MHz] $")
    #ax[i].set_ylabel(r"$\sigma_" + label[i] + r" [mm]$")
    ax[i].grid(linestyle="--")
    ax[i].legend(frameon=True)
#--

#
# sigma_y / sigma_z
# Simulated data
'''
plt.plot(delta, (std_dev_y / std_dev_z), label= "Simulation" , marker='o', linestyle='--')

# Plot paper data
plt.plot(p_delta_z, (p_std_dev_y / p_std_dev_z), label="Experiment", marker='^', linestyle='--')
'''

plt.show()
#--