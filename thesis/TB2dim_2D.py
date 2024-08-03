import matplotlib.pyplot as plt
import numpy as np

# Parameters
a = 2.461  # lattice constant (Å)

a1 = np.array([a * np.sqrt(3) / 2, a / 2])  # base vector 1
a2 = np.array([a * np.sqrt(3) / 2, -a / 2])  # base vector 2

E_2p = 0  # epsilon_2p, energy of level 2p (eV)
y0 = 2.7  # gamma_0, constant
s0 = 0.1  # s_0, constant

def energy_plus(kx, ky, a1, a2):
    """Calculates the higher energy band."""
    f = 3 + 2 * np.cos(kx * a1[0] + ky * a1[1]) + 2 * np.cos(kx * a2[0] + ky * a2[1]) + 2 * np.cos(kx * (a1[0] - a2[0]) + ky * (a1[1] - a2[1]))
    return (E_2p - y0 * np.sqrt(f)) / (1 - s0 * np.sqrt(f))

def energy_minus(kx, ky, a1, a2):
    """Calculates the lower energy band."""
    f = 3 + 2 * np.cos(kx * a1[0] + ky * a1[1]) + 2 * np.cos(kx * a2[0] + ky * a2[1]) + 2 * np.cos(kx * (a1[0] - a2[0]) + ky * (a1[1] - a2[1]))
    return (E_2p + y0 * np.sqrt(f)) / (1 + s0 * np.sqrt(f))

kx = np.linspace(-np.pi / a, np.pi / a, 100)
ky = np.linspace(-np.pi / a, np.pi / a, 100)
kx, ky = np.meshgrid(kx, ky)

E_plus = energy_plus(kx, ky, a1, a2)
E_minus = energy_minus(kx, ky, a1, a2)

plt.contourf(kx, ky, E_plus, cmap='viridis', alpha=0.7)
plt.contourf(kx, ky, E_minus, cmap='plasma', alpha=0.7)

plt.xlabel('kx (1/Å)')
plt.ylabel('ky (1/Å)')
plt.title('2D Projection of Energy Bands Structure of a 2D Atom Grid')
plt.colorbar(label='Energy (eV)')
plt.show()
