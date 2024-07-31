import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Physical constants
hbar = 6.582119569e-16  # Reduced Planck constant (eV·s)
m_e = 9.10938356e-31  # Electron mass (kg)
a = 5.431  # Silicon lattice constant (Å)
N = 100  # Number of points in reciprocal space

# High-symmetry points in the FCC Brillouin zone
Gamma = np.array([0, 0, 0])
X = np.array([2*np.pi/a, 0, 0])
W = np.array([2*np.pi/a, np.pi/a, 0])
K = np.array([3*np.pi/(2*a), 3*np.pi/(2*a), 0])
L = np.array([np.pi/a, np.pi/a, np.pi/a])

# Path in the Brillouin zone
k_path = np.concatenate([
    np.linspace(Gamma, X, N//4, endpoint=False),
    np.linspace(X, W, N//4, endpoint=False),
    np.linspace(W, K, N//4, endpoint=False),
    np.linspace(K, L, N//4)
])

# Function to calculate the pseudopotential
def pseudopotential(G):
    # Silicon pseudopotential Fourier components (in eV)
    V_G = {
        0: -12.0,
        3: 8.0,
        8: 4.0,
        11: 4.0
    }
    G_squared = np.dot(G, G)
    return V_G.get(G_squared, 0)

# Function to calculate the Hamiltonian matrix
def hamiltonian(k):
    G_vectors = [np.array([i, j, k]) * 2 * np.pi / a for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]
    H = np.zeros((len(G_vectors), len(G_vectors)), dtype=complex)
    for i, G1 in enumerate(G_vectors):
        for j, G2 in enumerate(G_vectors):
            if np.array_equal(G1, G2):
                H[i, j] = hbar**2 * np.dot(k + G1, k + G1) / (2 * m_e)
            H[i, j] += pseudopotential(G1 - G2)
    return H

# Calculate the band structure
band_structure = []
for k in k_path:
    H = hamiltonian(k)
    eigenvalues, _ = eigh(H)
    band_structure.append(eigenvalues[:6])  # Only take the first 6 bands

# Convert band structure to numpy array
band_structure = np.array(band_structure)

# Plot the band structure
for i in range(band_structure.shape[1]):
    plt.plot(band_structure[:, i], label=f'Band {i+1}')
plt.xlabel('k')
plt.ylabel('Energy (eV)')
plt.title('Band Structure of Silicon')
plt.legend()
plt.grid(True)
plt.show()
