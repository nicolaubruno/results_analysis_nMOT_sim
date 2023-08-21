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
# G = 1.71 G / cm
#--
p_delta = [-2.36,-2.15,-1.94,-1.72,-1.50,-1.30,-1.08,-0.87,-0.66,-0.45,-0.24,-0.14]
p_z_c = np.array([-0.0094,-0.0084,-0.0072,-0.0062,-0.0052,-0.0042,-0.0031,-0.0022,-0.0014,-0.0009,-0.0008,-0.0007])*1e2
del_B = "1.71"
#--

# Paper data
# G = 2.26 G / cm
'''
p_delta = [-2.5,-2.33,-2.15,-1.97,-1.80,-1.61,-1.44,-1.27,-1.09,-0.91,-0.73,-0.55,-0.37]
p_z_c = np.array([-0.0074,-0.0066,-0.0060,-0.0054,-0.0049,-0.0042,-0.0036,-0.0029,-0.0023,-0.0018,-0.0014,-0.0009,-0.0007])*1e2
del_B = "2.26"
'''
#
# Simulation data
#--
# Small detunings
#s_res = Results(1616642739)
#s_z_c, s_std_z_c = s_res.mass_centre(axis=[2])

# Middle detunings
#m_res = Results(1616088824)
#m_z_c, m_std_z_c = m_res.mass_centre(axis=[2])

# Large detunings
#l_res = Results(1616591691)
#l_z_c, l_std_z_c = l_res.mass_centre(axis=[2])

#
# Unique range
#--

#
# G = 2.26 G / cm
#--
#res = Results(1616843171)
#--

#
# G = 1.71 G / cm
#--
#res = Results(1616687235)
#res = Results(1616805096)
res = Results(1617226321)
#--

# Values
z_c, std_z_c = res.mass_centre(axis=[2])

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

#
# Theoretical values
#--
def z_0(delta):
    h = 6.626e-34
    u = 1.661e-27
    mu_B = 927.4e-26
    g = 9.81
    g_gnd = 1.24
    g_exc = 1.29
    m_gnd = -8
    m_exc = -9
    wavelength = 626e-9
    gamma = 2*np.pi*136e3
    M = 163.929174751*u
    s_0 = 0.65
    B_0 = 1.71e-2

    beta = (mu_B * 2 * np.pi * B_0 * (g_gnd * m_gnd - g_exc * m_exc)) / (2*h)

    return (2*delta + gamma * np.sqrt((h * gamma * s_0) / (2 * M * wavelength * g) - 1 - s_0)) / (2*beta)

t_z_c = np.array([z_0(d * 2 * np.pi * 1e6)*1e2 for d in delta])
#--
#--

#
# Set plot
#--
plt.clf()
plt.style.use('seaborn-whitegrid')
plt.subplots_adjust(left=0.3, right=0.7, top=0.85, bottom=0.15)
plt.rcParams.update({
        "font.size":12,\
        "axes.titlepad":14
    })
#--

#
# Set plot figure
fig = plt.figure(figsize=(5,4))
#gs = fig.add_gridspec(1, 2, wspace=0)
#ax = gs.subplots(sharey=True)
fig.subplots_adjust(top=0.75)
fig.suptitle("Centre of mass as a function of \nthe laser detuning " + r"($\nabla B = " + del_B + " G/cm$)", size=16, y=0.95)
#fig.suptitle("Ratio of standard deviations of \nthe laser detuning " + r"($\nabla B = " + del_B + " G/cm$)", size=16, y=0.95)
plt.close(1)

# Set labels
plt.xlabel(r"$ \Delta\ [2\pi \times MHz] $")
plt.ylabel(r"$z_0\ [cm]$")
#--

# Simulated data
plt.plot(delta, z_c, label='Simulation', marker='o', linestyle='--')

# Plot paper data
plt.plot(p_delta, p_z_c, label="Experiment", marker='^', linestyle='--')

# Theoretical values
#plt.plot(delta, t_z_c, label='Theoretical', color="black", marker='', linestyle='-')

# Set plot
plt.grid(linestyle="--")
plt.legend(frameon=True)

plt.show()
#--