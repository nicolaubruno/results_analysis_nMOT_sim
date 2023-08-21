#
# Libraries
#--
import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import newton, brentq
from Results import Results
#--

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

#
# Simulation data
#--
res = Results(1617610575)
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
#s = lambda opt: 2 * ((2 + opt*np.sqrt(2))/4)**2 * 0.65
w = 2.0 # 1e-2 m
chi = lambda transition: g_exc * (m_gnd + transition) - g_gnd * m_gnd
B_0 = 1.71 #  1e-2 T / m
beta = (np.pi * mu_B * B_0) / h # 1e8 m / s
M = 163.929174751*u # 1e-27 kg
wavelength = 626 # 1e-9 m
gamma = 2 * np.pi * 136 # 1e3 Hz
g_gnd = 1.24
g_exc = 1.29
m_gnd = -8
max_scatt_force = 1e-22 * (h * gamma) / (2 * wavelength)
#--

#
# Saturation parameter
# 
# Parameters:
#--
# z [cm]
# side [+1 or -1]
#
# Return
#--
# s_0 [dimensionless]
#--
def s(z, side):
    rho = (z * np.sqrt(2)) / 2
    s_0 = 2 * 0.65 * np.exp(-2 * (rho / w)**2)
    s_0 *= ((2 + side*np.sqrt(2))/4)**2

    return s_0   
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
def scatt_force(z, delta, direction):
    detuning = 2*np.pi*1e3*delta + 1e3*beta * chi(+1)*z
    F1 = (s(z, -direction) / (1 + s(z, -direction) + 4 * (detuning/gamma)**2))
    
    detuning = 2*np.pi*1e3*delta + 1e3*beta * chi(-1)*z
    F2 = (s(z, direction) / (1 + s(z, direction) + 4 * (detuning/gamma)**2))
    
    detuning = 2*np.pi*1e3*delta + 1e3*beta * chi(0)*z
    F3 = (s(z, 0) / (1 + s(z, 0) + 4 * (detuning/gamma)**2))

    return direction*(F1 + F2 + F3)
#--

print(scatt_force(1, 1, 1))
exit(0)

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
#--

#
# Harmonic oscillation constant [1e-20]
#
# Parameters:
#--
# z [cm]
# delta [2pi x MHz]
#
# Return:
#--
# alpha [1e-22]
#--
def alpha(z, delta, sign=+1):
    n = 4 * h * s(z, sign) * (delta*2*np.pi - beta*chi(sign)*z)
    d = wavelength * gamma * (1 + s(z, sign) + 4 * (1e3*(delta*2*np.pi + beta*chi(+1)*z)/gamma)**2)**2
    return (n / d)
#--

#
# Approximated equilibrium point
#
# Parameters:
#--
# delta [2pi x MHz]
#
# Return: 
#--
# z_0 [cm]
#--
def z_0_approx(delta):
    z_0 = 1e5 * (h*gamma*s(0, 1))/(2*M*wavelength*g)
    z_0 += -(1 + s(0, 1))
    z_0 = np.sqrt(z_0)
    z_0 *= 1e3*(gamma / 2)
    z_0 += 2 * np.pi * 1e6 * delta
    z_0 *= - 1e-8 / (beta*chi(-1))

    return z_0*1e2
#--

#
# Emanuel's results
#
# Parameters:
#--
# delta [2pi x MHz]
#
# Return: 
#--
# z_0 [cm]
#--
def z_0_emanuel(delta):
    s_0 = 0.65
    beta = 2 * np.pi * 8 * (h*1e-34) * 2*np.pi*delta*1e6 * s_0
    beta = beta / (gamma*1e3 * (wavelength*1e-9)**2 * (1 + s_0 + 4*(2*np.pi*delta*1e3 / gamma)**2)**2)
    print(delta, beta)
    #beta = (2 * np.pi)**2 * 8 * (h / (wavelength)**2) * delta * s_0
    #beta = beta / (gamma*(1 + s_0 + 4 * (2*np.pi*delta*1e3 / gamma)**2)**2)**2
    z_0 = 1e-27 * M * g / beta 
    return z_0*1e2
#--

#
# Function to minimize
#--
def f(z, delta):
    return scatt_force(z, delta, +1) + scatt_force(z, delta, -1) - (M*g*1e-27 / max_scatt_force) + magnetic_force(z)
    #return scatt_force(z, delta, -1) - (M*g*1e-27 / max_scatt_force)

def f_prime(z, delta):
    return (1e-14 / max_scatt_force) * beta*(alpha(z, delta, +1) * chi(+1) - alpha(z, delta, -1)*chi(-1))
    #return (1e-14 / max_scatt_force) * beta*(alpha(z, delta, -1)*chi(-1))
#--

#
# Theoretical data
t_z_c = np.zeros(len(delta))
for i, d in enumerate(delta[:]):
    z = z_c[i]
    delta_z = 0.2
    t_z_c[i] = brentq(f, z, z + delta_z, args=(d,))
    print(s(0, +1), s(0, -1))
    print(s(t_z_c[i], +1), s(t_z_c[i], -1))
    print()

#exit(0)

#
# Visualization
#--

#
# Set plot
#--
plt.clf()
plt.style.use('seaborn-whitegrid')
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.rcParams.update({
        "figure.figsize": (6,5),\
        "font.size":12,\
        "axes.titlepad":14
    })
#--

# Set labels
plt.title("Centre of mass as a function of \nthe laser detuning " + r"($\nabla B = " + del_B + " G/cm$)")
plt.xlabel(r"$ \Delta\ [2\pi \times MHz] $")
plt.ylabel(r"$z_0\ [cm]$")
#--
#--

#
# Plots
#--
# Theoretical 2 beams
plt.plot(delta, t_z_c, label="Bruno's Calculus", marker='s', linestyle='--')

# Theoretical 1 beams
plt.plot(delta, [z_0_approx(d) for d in delta], label="Larissa's Calculus", marker='o', linestyle='--')

# Plot paper data
plt.plot(p_delta, p_z_c, label="Experiment", marker='^', linestyle='--')

# Simulated data
plt.plot(delta, z_c, label="Simulation", marker='*', linestyle='--')

# Theoretical values
#plt.plot(delta, [z_0_emanuel(d) for d in delta], label="Emanuel\'s calculus", marker='s', linestyle='--')
#--

# Set plot
plt.grid(linestyle="--")
plt.legend(frameon=True)

plt.show()
#--