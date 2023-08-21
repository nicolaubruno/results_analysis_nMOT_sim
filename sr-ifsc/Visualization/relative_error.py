#
# Libraries
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from Results import Results

#
# Paper data

#
# G = 1.71 G / cm
'''
p_delta = [-2.36,-2.15,-1.94,-1.72,-1.50,-1.30,-1.08,-0.87,-0.66,-0.45,-0.24,-0.14]
p_z_c = np.array([-0.0094,-0.0084,-0.0072,-0.0062,-0.0052,-0.0042,-0.0031,-0.0022,-0.0014,-0.0009,-0.0008,-0.0007])*1e2
del_B = "1.71"
'''

#
# G = 2.26 G / cm
#--
p_delta = [-2.5,-2.33,-2.15,-1.97,-1.80,-1.61,-1.44,-1.27,-1.09,-0.91,-0.73,-0.55,-0.37]
p_z_c = np.array([-0.0074,-0.0066,-0.0060,-0.0054,-0.0049,-0.0042,-0.0036,-0.0029,-0.0023,-0.0018,-0.0014,-0.0009,-0.0007])*1e2
del_B = "2.26"
#--

#
# Simulation data
#--
#s_res = Results(1616642739)
#s_z_c, s_std_z_c = s_res.mass_centre(axis=[2])

#m_res = Results(1616088824)
#m_z_c, m_std_z_c = m_res.mass_centre(axis=[2])

#l_res = Results(1616591691)
#l_z_c, l_std_z_c = l_res.mass_centre(axis=[2])

#
# G = 2.26 G / cm
#--
#res = Results(1616693405)
res = Results(1616843171)
#--

#
# G = 1.71 G / cm
'''
res = Results(1616687235)
'''

# Values
z_c, std_z_c = res.mass_centre(axis=[2])
#--

#
# Visualization
#
# Clear stored plots
plt.clf()

#
# Set style
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
        "figure.figsize": (12,10),\
        "font.size":12,\
        "axes.titlepad":14
    })

plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

style={
    "linestyle":'-'
}

markers = ['o', '^', 's']

#
# Set labels
labels = ['x', 'y', 'z']
plt.title("Relative error of the simulated centre of mass\n related to the experiment values " r"($\nabla B = " + del_B + " G/cm$)")
plt.xlabel(r"$ \Delta\ [2\pi \times MHz] $")
plt.ylabel(r"$|z_{experiment} - z_{simulation}|\ [mm]$")

#s_delta = np.array(s_res.loop["values"])*(s_res.transition["gamma"]*1e-3)
#m_delta = np.array(m_res.loop["values"])*(m_res.transition["gamma"]*1e-3)
#l_delta = np.array(l_res.loop["values"])*(l_res.transition["gamma"]*1e-3)
#delta = np.concatenate((s_delta, m_delta), axis=None)
#z_c = np.concatenate((s_z_c, m_z_c), axis=None)
#std_z_c = np.concatenate((s_std_z_c, m_std_z_c), axis=None)

delta = np.array(res.loop["values"])*(res.transition["gamma"]*1e-3)
#delta = np.array(s_res.loop["values"])*(s_res.transition["gamma"]*1e-3)
#z_c = s_z_c
#std_z_c = s_std_z_c

#
# Simulated data
plt.plot(p_delta, np.abs(p_z_c - z_c[::-1])*10, marker='o', linestyle='--')

#
# Set plot
#plt.xlim([np.min(delta)*1.1, np.max(delta)*0.9])
plt.grid(linestyle="--")
#plt.legend(frameon=True)

#plt.xlim([delta[-1], delta[0]])

#
# Show
plt.show()