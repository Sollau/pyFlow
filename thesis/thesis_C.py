import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Importing tqdm for the progress bar

a = 5.431  # Lattice constant of Silicon (in Å)
N = 100  # Number of points in the reciprocal space
tau = np.array([1, 1, 1]) * (a / 8)  # Primitive vector

h_bar = 6.582e-16  # Reduced Planck constant (in eV*s)
m = 0.511e6  # Mass of the electron (in eV/c**2)

c = 3e8  # Light speed (in m/s)

conv = 27.2114  # Conversion factor from Eh to eV


def V_ps(G):
    """Fourier components (form factors) of the pseudopotential in eV."""
    if np.isclose(np.dot(G, G), 3 * (2 * np.pi / a) ** 2, atol=1e-3):
        return -0.1121 * conv
    elif np.isclose(np.dot(G, G), 8 * (2 * np.pi / a) ** 2, atol=1e-3):
        return 0.0276 * conv
    elif np.isclose(np.dot(G, G), 11 * (2 * np.pi / a) ** 2, atol=1e-3):
        return 0.0362 * conv
    else:
        return 0


def V_ps_TOT(G):
    """Total pseudopotential, given a vector of the 1st Brillouin zone G."""
    structure_factor = np.cos(np.dot(G, tau))  # Structure factor
    V = V_ps(G) * structure_factor  # Total pseudopotential
    return V


def T(G, k):
    """Calculates the kinetic energy (in eV), given a vector of the 1st Brillouin zone G and a k vector."""
    T = (
        ((h_bar*c)**2 / (2 * m *1E-20)) * (np.linalg.norm(k + G)) **2  # Add 1E-20 factor for conversion from Å**2 to m**2
    )
    return T


n = 3  # Regulates the ranges of G_vectors in H_fill


def H_fill(k):
    """Calculates the Hamiltonian matrix in the 1st Brillouin zone, given k vectors."""
    G_vectors = np.array(
        [[i, j, l] for i in range(-n, n) for j in range(-n, n) for l in range(-n, n)]
    ) * (2 * np.pi / a)
    H = np.zeros((len(G_vectors), len(G_vectors)), dtype=np.float64)
    for i in range(len(G_vectors)):
        H[i][i] += T(G_vectors[i], k)
        for j in range(len(G_vectors)):
            G = G_vectors[i] - G_vectors[j]
            H[i][j] += V_ps_TOT(G)
    return H


if __name__ == "__main__":
    # High-symmetry points (in 1/Å)
    Gamma = np.array([0, 0, 0])
    X = np.array([2 * np.pi / a, 0, 0])
    W = np.array([2 * np.pi / a, np.pi / a, 0])
    K = np.array([3 * np.pi / (2 * a), 3 * np.pi / (2 * a), 0])
    L = np.array([np.pi / a, np.pi / a, np.pi / a])

    # Irreducible 1st Brillouin zone boundary path
    k_path = np.concatenate(
        [
            np.linspace(Gamma, X, 24, endpoint=False),
            np.linspace(X, W, 12, endpoint=False),
            np.linspace(W, L, 17, endpoint=False),
            np.linspace(L, Gamma, 21, endpoint=False),
            np.linspace(Gamma, K, 26),
        ]
    )

    # Energy eigenvalues
    E = []
    for k in tqdm(k_path, desc="Calculating energy eigenvalues"):  # Progress bar
        e = np.linalg.eigvalsh(H_fill(k))  # Computes eigenvalues e
        E.append(e)

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, band in enumerate(np.array(E).T[:7]):
        plt.plot(np.linspace(0, 1, len(k_path)), band, label=f"Band {i+1}")

    # Add vertical lines for high-symmetry points
    high_symmetry_points = [0, 24, 36, 53, 74, 100]
    labels = ["Γ", "X", "W", "L", "Γ", "K"]
    for point in high_symmetry_points:
        plt.axvline(x=point / len(k_path), color="k", linestyle="--")
    plt.xticks([point / len(k_path) for point in high_symmetry_points], labels)

    plt.xlabel("k-path (Å)")
    plt.ylabel("Energy ($eV$)")
    plt.title("Band Structure of Silicon")
    plt.legend()
    plt.grid(True)
    plt.show()
