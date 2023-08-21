#
# Libraries
#--
import numpy as np
import cmath

from matplotlib import pyplot as plt
from scipy.optimize import newton, brentq, root_scalar
from Results import Results
#--

#
# Data
#--
#
# Paper data
# G = 1.71 G / cm
#--
p_delta = [-1.72,-1.50,-1.30,-1.08,-0.87]
p_z_c = np.array([-0.0062,-0.0052,-0.0042,-0.0031,-0.0022])*1e2
del_B = "1.71"
#--

#
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
res = Results(1624296860) # del_B = 1.71 G / cm
#res = Results(1616843171) # del_B = 2.26 G / cm
z_c, std_z_c = res.mass_centre(axis=[2])
delta = np.array(res.loop["values"])*(res.transition["gamma"]*1e-3)
#--

#
# Physical constants
#--
h = 6.626070040 # 1e-34 J s
u = 1.660539040 # 1e-27 kg
g = 9.81 # m / s^2
mu_B = 9.274009994 # 1e-24 J / T
#--

#
# Parameters
#--
s_0 = 0.65
w = 2.0 # 1e-2 m
chi = lambda transition: g_exc * (m_gnd + transition) - g_gnd * m_gnd
B_0 = float(del_B) #  1e-2 T / m
beta = (np.pi * mu_B * B_0) / h # 1e8 m / s
m = 163.929174751*u # 1e-27 kg
wavelength = 626 # 1e-9 m
gamma = 2 * np.pi * 136 # 1e3 Hz
g_gnd = 1.24
g_exc = 1.29
m_gnd = -8
max_scatt_force = 1e-22 * (h * gamma) / (2 * wavelength)
#--

#
# Change-of-basis matrices
#--
def R(theta):
    R = np.zeros((3,3), dtype=complex)
    R[0][0] = complex(1)
    R[1][1] = complex(np.cos(theta))
    R[1][2] = complex(-np.sin(theta))
    R[2][1] = complex(np.sin(theta))
    R[2][2] = complex(np.cos(theta))

    return R

#
# Change of basis from Cartesian basis to polarization basis
#--
M = np.zeros((3,3), dtype=complex)

M[0][0] = complex(1 / np.sqrt(2))
M[0][1] = 1j / complex(np.sqrt(2))

M[1][0] = complex(1 / np.sqrt(2))
M[1][1] = -1j / complex(np.sqrt(2))

M[2][2] = 1
#--
#--

#
# Polarization on the lab frame basis
# 
# Parameters:
#--
# pol [3D-array] (Polarization on the beam frame)
# theta [float]
#
# Return
#--
# eps [3D-array]
#--
def eps(pol, theta):
    A = np.dot(M, np.dot(R(theta), np.conjugate(M.T)))
    return np.dot(A, pol)
#--

#
# Saturation parameter
# 
# Parameters:
#--
# z [cm]
# theta [radian] (Angle of the wave vector with respect to y-axis)
# component [integer] (Polarization component: 0 -> sigma+, 1 -> sigma-, 2 -> pi)
#
# Return
#--
# s_0 [dimensionless]
#--
def s(z, theta, component):
    s = s_0 * np.exp(-2 * (np.abs(np.sin(theta)*z) / w)**2)
    s *= np.abs(eps((0,1,0), theta)[component])**2

    return s
#--

#
# Scattering force
#
# Parameters:
#--
# z [cm]
# delta [2pi x MHz]
#
# Return:
#--
# F [max_scatt_force]
#--
def scatt_force(z, delta):
    F = 0
    transitions = [+1, -1, 0]
    all_theta = np.array([-1/4, +1/4, -3/4, 3/4])*np.pi

    for i, theta in enumerate(all_theta):
        direction = +1 if np.abs(theta) < (3*np.pi/4) else -1

        for j in range(3):
            s_j = s(z, theta, j)
            detuning = 2*np.pi*1e3*delta + 1e3*beta * z * chi(transitions[j])
            F_j = direction * (np.sqrt(2)/2) *(s_j / (1 + s_j + 4 * (detuning/gamma)**2))
            F += F_j

    return F
#--

#
# Magnetic force
#
# Parameters:
#--
# z [cm]
#
# Return:
#--
# F_B [max_scatt_force]
#--
def magnetic_force(z):
    del_B = B_0 * z / (2*abs(z))
    return - (1e-28 / max_scatt_force) * g_gnd * m_gnd * mu_B * del_B

#
# Theoretical values
#--
def f(z, delta):
    return scatt_force(z, delta) - (m*g*1e-27 / max_scatt_force) + magnetic_force(z)

t_z_c = np.zeros(len(delta))
for i, d in enumerate(delta):
    z = z_c[i]
    delta_z = 0.2
    bracket = [z - delta_z, z + delta_z]
    #print("z_guess =", z)
    #print("f(a) =", f(bracket[0], d))
    #print("f(b) =", f(bracket[1], d))
    t_z_c[i] = root_scalar(f, bracket=bracket, args=(d,), method="bisect").root
    #print("z_0 =", t_z_c[i], end="\n\n")

#
# Visualization
#--

#
# Set plot figure
#--
plt.clf()

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
        "font.size":12,\
        "axes.titlepad":10
    })

fig = plt.figure(figsize=(6,5))
plt.subplots_adjust(left=0.13, right=0.95, top=0.83, bottom=0.10)
gs = fig.add_gridspec(4, 1, hspace=0)
ax = [fig.add_subplot(gs[:3, 0]), fig.add_subplot(gs[3, 0])]
fig.suptitle("Centre of mass as a function of \nthe laser detuning " + r"($\nabla B = " + del_B + " G/cm$)", size=16, y=0.95)
plt.close(1)
#--

#
# Plot centre of mass
#--
# Theory
ax[0].plot(delta, t_z_c, label="Theory", marker='o', linestyle='--')

# Simulated data
ax[0].plot(delta, z_c, label="Simulation", marker='^', linestyle='--')

# Plot paper data
ax[0].plot(p_delta, p_z_c, label="Experiment", marker='s', linestyle='--')

# Settings
ax[0].set_ylabel(r"$z_0\ [cm]$")
ax[0].grid(linestyle="--")
ax[0].legend(frameon=True)
#--

#
# Plot error
#--
# Theory
ax[1].plot(delta, np.abs(t_z_c - p_z_c[::-1])*10, marker='o', linestyle='--')

# Simulation
ax[1].plot(delta, np.abs(z_c - p_z_c[::-1])*10, marker='^', linestyle='--')

# Settings
ax[1].set_ylim(0, 1.1)
ax[1].set_ylabel(r"$|z_{0} - z_{exp}|\ [mm]$")
ax[1].set_xlabel(r"$ \Delta\ [2\pi \times MHz] $")
ax[1].grid(linestyle="--")
#--

# Set plot

plt.tight_layout()
plt.show()
#--