import matplotlib.pyplot as plt
import numpy as np

# Parameters
a = 2.461  # lattice constant (Å)
a1 = np.array([a * np.sqrt(3) / 2, a / 2])  # base vector 1
a2 = np.array([a * np.sqrt(3) / 2, -a / 2])  # base vector 2

E_2p = 0  # epsilon_2p, energy of level 2p (eV)
y0 = 2.7  # gamma_0, constant
s0 = 0.1  # s_0, constant

def energy_plus(k, a1, a2):
    """Calculates the higher energy band."""
    f = 3 + 2 * np.cos(k * a1[0]) + 2 * np.cos(k * a2[0]) + 2 * np.cos(k * (a1[0] - a2[0]))
    return (E_2p - y0 * np.sqrt(f)) / (1 - s0 * np.sqrt(f))

def energy_minus(k, a1, a2):
    """Calculates the lower energy band."""
    f = 3 + 2 * np.cos(k * a1[0]) + 2 * np.cos(k * a2[0]) + 2 * np.cos(k * (a1[0] - a2[0]))
    return (E_2p + y0 * np.sqrt(f)) / (1 + s0 * np.sqrt(f))

k = np.linspace(-2*np.pi / a, 2*np.pi / a, 100)

E_plus = energy_plus(k, a1, a2)
E_minus = energy_minus(k, a1, a2)

# Graph
plt.plot(k, E_plus, label='Higher Energy Band')
plt.plot(k, E_minus, label='Lower Energy Band')
plt.xlabel('k (1/Å)')
plt.ylabel('Energy (eV)')
plt.title('1D Projection of Energy Bands Structure of a 2D Atom Grid')
plt.legend()
plt.show()
