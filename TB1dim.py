import numpy as np
import matplotlib.pyplot as plt

#Parameters
a = 2 #lattice pitch (in units of Bohr Radius [a0 = 0.53Å])
k_points = np.linspace(-np.pi/a, np.pi/a, 100) #k-points
V0 = -1  #Pseudopotential (eV)

def energy(k, V0, a):
    """Calculates the energy, given a k-point, the pseudopotential and the lattice pitch."""
    return V0 * (1 + np.cos(k * a))

#Energies array
energies = energy(k_points, V0, a)

#Graph
plt.plot(k_points, energies, label='Energy band')
plt.xlabel('k (1/Å)')
plt.ylabel('Energy (eV)')
plt.title('Energy Bands Structure of a 1D Atom Chain')
plt.legend()
plt.grid(True)
plt.show()
