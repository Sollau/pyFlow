import numpy as np
import matplotlib.pyplot as plt

a = 5.431  # Lattice constant of Silicon (in atomic units)
N = 100  # Number of points in the reciprocal space
tau = a * np.array([1/8, 1/8, 1/8]) # Primitive vector

def V_ps(G):
    """Total pseudopotential in atomic units, expanded in Fourier components."""
    if np.linalg.norm(G) == np.sqrt(3):
        return -0.1121
    elif np.linalg.norm(G) == np.sqrt(8):
        return 0.0276
    elif np.linalg.norm(G) == np.sqrt(11):
        return 0.0362
    return 0

def T(G, k):
    """Calculates the kinetic energy, given a vector of the 1st Brillouin zone G and a k vector."""
    return 0.5 * np.linalg.norm(k + G) ** 2

# High-symmetry points
Gamma = np.array([0, 0, 0])
X = np.array([2*np.pi/a, 0, 0])
W = np.array([2*np.pi/a, np.pi/a, 0])
K = np.array([3*np.pi/(2*a), 3*np.pi/(2*a), 0])
L = np.array([np.pi/a, np.pi/a, np.pi/a])

k_path = np.concatenate([
    np.linspace(Gamma, X, 24, endpoint=False),
    np.linspace(X, W, 12, endpoint=False),
    np.linspace(W, L, 17, endpoint=False),
    np.linspace(L, Gamma, 21, endpoint=False),
    np.linspace(Gamma, K, 26)
])

def H_fill(k):
    """Calculates the Hamiltonian matrix in the 1st Brillouin zone, given k vectors."""
    G_vectors = np.array([[i, j, l] for i in range(-4, 4) for j in range(-4, 4) for l in range(-4, 4)]) * (2 * np.pi / a)
    H = np.zeros((len(G_vectors), len(G_vectors)), dtype=np.float64)
    for i in range(len(G_vectors)):
        H[i][i] += T(G_vectors[i], k)
        for j in range(len(G_vectors)):
            G = G_vectors[i] - G_vectors[j]
            H[i][j] += V_ps(G) * np.cos(np.dot(G, tau) * np.pi / 4)
    return H

# Energy eigenvalues
E = []
for k in k_path:
    e, _ = np.linalg.eigh(H_fill(k))  # Extract eigenvalues e 
    E.append(e)

# Plotting
plt.figure(figsize=(10, 6))
for i, band in enumerate(np.array(E).T[:5]):
    plt.plot(np.linspace(0, 1, len(k_path)), band, label=f'Band {i+1}')

# Adding vertical lines for high-symmetry points
high_symmetry_points = [0, 24, 36, 53, 74, 100]
labels = ['Γ', 'X', 'W', 'L', 'Γ', 'K']
for point in high_symmetry_points:
    plt.axvline(x=point/len(k_path), color='k', linestyle='--')
plt.xticks([point/len(k_path) for point in high_symmetry_points], labels)

plt.xlabel('k-path')
plt.ylabel('Energy (eV)')
plt.title('Band Structure of Silicon')
plt.legend()
plt.grid(True)
plt.show()
