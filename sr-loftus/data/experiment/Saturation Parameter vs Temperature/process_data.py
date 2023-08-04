# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = pd.read_csv("simulated_data.csv", header=0)
paper_data = pd.read_csv("paper_data.csv", header=0)

# Regime (II): Theoretical
gamma = 2 * np.pi * 7.5e3 # Hz
T = lambda s: ((6.626070040 * gamma * np.sqrt(1 + s)) / (2 * 2 * np.pi * 1.38064852)) * 1e-5 # mu K
s = np.linspace(np.min(paper_data.iloc[:,0]), np.max(paper_data.iloc[:,0]), 1000)


# Fitting
'''
final_idx = -5

# Simulated data
A = np.vstack([data.iloc[:final_idx,0], np.ones(len(data.iloc[:final_idx,0]))]).T
m1, c1 = np.linalg.lstsq(A, data["z0 [cm]"].to_numpy()[:final_idx], rcond=None)[0]

# Paper data
A = np.vstack([paper_data.iloc[:final_idx,0]/(-7.5e-3), np.ones(len(paper_data.iloc[:final_idx,0]))]).T
m2, c2 = np.linalg.lstsq(A, paper_data.iloc[:,1].to_numpy()[:final_idx]*0.1, rcond=None)[0]
'''

# Plot
'''
x = np.linspace(np.min(data.iloc[:final_idx,0]), np.max(data.iloc[:final_idx,0]), 1000)
y1 = m1*x + c1
y2 = m2*x + c2
m_theory = ((1.0545718e-10)/(9.274009994 * 10e-2 * 1.5)) * (2 * np.pi * 7.5e3) * 1e2
'''

plt.clf()
plt.rcParams.update({"font.size":14})
plt.plot(data.iloc[:,0], data.iloc[:,1], marker="o", linestyle="", label="Simulated Data")
#plt.plot(x, y1*10, marker="", linestyle="--", color="black", label=("Fitting \n(Slope Error: %.1f" % (100*(1 - m_theory / m1))) + r"%")
plt.plot(paper_data.iloc[:,0], paper_data.iloc[:,1], marker="o", linestyle="", label="Experimental Data")
plt.plot(s, [T(si) for si in s], marker="", linestyle="--", color="black", label="Theoretical Curve")
#plt.plot(x, y2*10, marker="", linestyle="-", color="black", label=("Fitting \nSlope Error: %.1f" % (100*(1 - m_theory / m2))) + r"%")

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$s$")
plt.ylabel(r"$T [\mu K]$")

plt.tight_layout()
plt.legend(frameon=True)
plt.grid(linestyle="--")
plt.show()