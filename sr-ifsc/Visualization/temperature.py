#
# Libraries
#--
import numpy as np
import matplotlib.pyplot as plt

from Results import Results
#--

#
# Data
#--
#
# Dreon paper
#p_delta = -np.array([0.01, 0.35, 0.70, 1.05, 1.40, 1.76, 2.10, 2.25, 2.32, 2.39, 2.43, 2.46])[::-1] / 0.136 # delta / gamma
p_delta = -np.array([0.01, 0.35, 0.70, 1.05, 1.40, 1.76, 2.10, 2.25, 2.32, 2.39, 2.43, 2.46])[::-1] # 2*pi*MHz
p_temp = np.array([20.67, 20.86, 21.05, 21.23, 19.92, 20.10, 16.28, 16.62, 18.63, 28.66, 41.35, 58.14]) # uK

#
# Hanley paper
'''
p_delta = np.array([-400.72262512844185, -300.8495778171583, -200.5554645500439, -149.99739393307624, -100.28815654271716, -70.37236973388343, -50.1399830233355]) / 7.5  # delta
p_temp = np.array([1.8320441988950276, 1.816574585635359, 1.7679558011049723, 1.7303867403314916, 1.7723756906077348, 1.7790055248618786, 1.7458563535911602]) # uK

temp = np.array([1.8784530386740332, 1.8784530386740332, 1.8696132596685082, 1.867403314917127, 1.867403314917127, 1.8585635359116022, 1.8386740331491713, 1.8320441988950276, 1.7812154696132596, 1.7701657458563536, 1.7591160220994475, 1.752486187845304, 1.7414364640883977, 1.6685082872928176, 1.6596685082872926, 1.569060773480663, 1.3215469613259667, 1.202209944751381, 0.9701657458563533]) # uK
delta = np.array([-401.15039239921964, -320.69217137496094, -280.6607496537654, -380.92917454691667, -340.4979077005555, -360.70907358043814, -260.8337924975056, -300.4530833494661, -200.5621658649908, -220.36790219058543, -160.5240428288484, -240.57460052717013, -180.32642849696958, -120.45129633214697, -140.258149543566, -100.58971571532817, -80.24899107980525, -59.973045822102506, -40.04445205581459]) / 7.5 # delta / gamma
'''

#
# G = 1.71 G / cm
#res = Results(1616979004)
#res = Results(1622296705)
temp = res.temperature(method=1)*1e6 #uK
delta = np.array(res.loop["values"])*0.136
#--

#
# Plot
#--
# Clear stored plots
plt.clf()

# Set figure
plt.figure(figsize=(5,4))
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size':14})
#fig.subplots_adjust(top=0.75)

# Set labels
plt.title('Temperature as a function of laser detuning', y=1.05)
plt.xlabel(r"$\Delta [2\pi \times MHz]$")
#plt.xlabel(r"$\Delta / \Gamma$")
plt.ylabel(r"$T [\mu K]$")

# Plot simulated data
plt.plot(delta, temp, label="Simulation", marker='o', linestyle="")

# Plot paper data
plt.plot(p_delta, p_temp, label="Experiment", marker='^', linestyle="--")

# Set grid and legend
plt.grid(linestyle="--")
plt.legend(frameon=True)

plt.close(1)
plt.show()
#--
