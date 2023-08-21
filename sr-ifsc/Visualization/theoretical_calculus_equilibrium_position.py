#
# Libraries
#--
import numpy as np

from scipy.optimize import newton, brentq
from matplotlib import pyplot as plt
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
res = Results(1617226321)
z_c, std_z_c = res.mass_centre(axis=[2])
delta = np.array(res.loop["values"])*(res.transition["gamma"]*1e-3)
#--
#--

#
# Parameters
#--
# Physical constant
h = 6.626 # 1e-34 J s
u = 1.661 # 1e-27 kg
mu_B = 9.274 # 1e-24 J / T

# Environment
s_0 = 0.65
B_0 = 1.71 # 1e-2 G / cm
wavelength = 626 # nm
g = 9.81 # m / s
beta = (np.pi * mu_B * B_0) / h # 1e8

# Atom and transition
M = 163.929174751*u # 1e-27 kg
gamma = 2*np.pi*136 # kHz
g_exc = 1.29
g_gnd = 1.24
m_gnd = -8
m_plus = -7
m_minus = -9
#--

#
# Larissa's equation
#--
def z_0_guess(delta):
    beta = ((mu_B*1e-24) * 2 * np.pi * (B_0*1e-2) * (g_gnd * m_gnd - g_exc * m_minus)) / (2*(h*1e-34))
    return (2*delta + (gamma*1e3) * np.sqrt(((h*1e-34) * (gamma*1e3) * s_0) / (2 * (M*1e-27) * (wavelength*1e-9) * g) - 1 - s_0)) / (2*beta)
#--

#
# Chi
#--
def chi(transition):
    chi = - g_gnd * m_gnd

    if transition == +1:
        chi += g_exc * m_plus
    else:
        chi += g_exc * m_minus

    return chi
#--

#
# Scattering force at equilibrium [1e-22 N]
#--
def scatt_force(z, delta, transition=+1):
    return transition*(h * gamma * s_0) / (2 * wavelength * (1 + s_0 + 4 * ((delta*1e-3 - (beta*1e5) * chi(transition) * z)/gamma)**2))
#--

#
# Harmonic oscillation constant [1e-20]
#--
def alpha(z, delta, sign=+1):
    n = 4 * h * s_0 * (delta*1e-8 - beta*chi(sign)*z)
    d = wavelength * gamma * (1 + s_0 + 4 * ((delta*1e-3 - (beta*1e5)*chi(+1)*z)/gamma)**2)**2
    return (n / d)
#--

#
# Function to find roots
#--
def f(z, delta):
    #return (scatt_force(z, delta, +1) + scatt_force(z, delta, -1) - 1e-5 * M * g)
    return (scatt_force(z, delta, +1) - M * g * 1e-5)

def f_prime(z, delta):
    #return (beta*1e8)*(alpha(z, delta, +1) * chi(+1) - alpha(z, delta, -1)*chi(-1))
    return 1e10 * beta*(alpha(z, delta, +1) * chi(+1))

#--

#
# Get theoretical results
#--
t_z_c = np.zeros(len(delta))

for i, d in enumerate((2*np.pi*delta*1e6)):
    z = z_c[i]*1e-2
    delta_z = 0.7e-2
    print("d =", d / (2*np.pi*1e6))
    print("f(a) =", f(z - delta_z, d))
    print("f(b) =", f(z + delta_z, d))
    t_z_c[i] = 1e2*brentq(f, z - delta_z, z + delta_z, args=(d,))
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
plt.rcParams.update({
        "figure.figsize": (12,10),\
        "font.size":12,\
        "axes.titlepad":14
    })

plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

# Set labels
plt.title("Centre of mass as a function of \nthe laser detuning " + r"($\nabla B = " + del_B + " G/cm$)")
plt.xlabel(r"$ \Delta\ [2\pi \times MHz] $")
plt.ylabel(r"$z_0\ [cm]$")
#--

# Simulated data
plt.plot(delta, z_c, label='Simulation', marker='o', linestyle='--')

# Plot paper data
plt.plot(p_delta, p_z_c, label="Experiment", marker='^', linestyle='--')

# Theoretical values
plt.plot(delta, t_z_c, label='Theoretical', marker='s', linestyle='--')

# Theoretical values
plt.plot(delta, [z_0_guess(2*np.pi*d*1e6)*1e2 for d in delta], label='Theoretical', marker='s', linestyle='--')

# Set plot
plt.grid(linestyle="--")
plt.legend(frameon=True)

plt.show()
#--
