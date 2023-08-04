# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = pd.read_csv("data.csv")

# Fitting
A = np.vstack([data.iloc[:-4,0], np.ones(len(data.iloc[:-4,0]))]).T
m, c = np.linalg.lstsq(A, data["z0 [cm]"].to_numpy()[:-4], rcond=None)[0]

# Plot
x = np.linspace(np.min(data.iloc[:-4,0]), np.max(data.iloc[:-4,0]), 1000)
y = m*x + c
m_theory = ((2 * 1.0545718e-10)/(9.274009994 * 1.71e-2 * (1.29*9 - 1.24*8))) * (2 * np.pi * 136e3)

#print(m_theory, m)
#exit(0)

plt.clf()
plt.rcParams.update({"font.size":12})
plt.plot(data.iloc[:,0], data["z0 [cm]"].to_numpy(), marker="o", linestyle="", label="Simulated Data")
plt.plot(x, y, marker="", linestyle="--", color="black", label="Fitting")
plt.legend(frameon=True)
plt.grid(linestyle="--")
plt.show()

