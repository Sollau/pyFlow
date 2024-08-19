import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Importing tqdm for the progress bar
import scipy.constants as const

a = 5.431  # Lattice constant of Silicon (in Å)
N = 100  # Number of points in the reciprocal space
tau = np.array([1, 1, 1]) * (a / 8)  # Primitive vector of Bravais lattice (in Å)

J_to_eV = 6.2415e18
Ry_to_eV = 13.6057039763

choice = str(input("Do you want to pass through U or K in k-path? [u / k]: "))


def V_ps(G):
    """Fourier components (form factors) of the pseudopotential in eV."""
    if np.isclose(np.dot(G, G), 3 * (2 * np.pi / a) ** 2, atol=1e-3):
        return -0.2241 * Ry_to_eV
    elif np.isclose(np.dot(G, G), 8 * (2 * np.pi / a) ** 2, atol=1e-3):
        return 0.0551 * Ry_to_eV
    elif np.isclose(np.dot(G, G), 11 * (2 * np.pi / a) ** 2, atol=1e-3):
        return 0.0724 * Ry_to_eV
    else:
        return 0


def V_ps_TOT(G):
    """Total pseudopotential, given a vector of the 1st Brillouin zone G."""
    S = np.cos(np.dot(G, tau))  # Structure factor (dimensionless)
    V = V_ps(G) * S  # Total pseudopotential (Ry)
    return V


def T(G, k):
    """Calculates the kinetic energy (in eV), given a vector of the 1st Brillouin zone G and a k vector."""
    T = (
        (const.hbar**2 / (2 * const.m_e))
        * ((1e10 * np.linalg.norm(k + G)) ** 2)
        * J_to_eV
    )
    return T


n = 3  # Regulates the ranges of G_vectors in H_fill


def H_fill(k):
    """Calculates the Hamiltonian matrix in the 1st Brillouin zone, given k vectors."""
    G_vectors = np.array(
        [
            [i, j, l]
            for i in range(-n, n + 1)
            for j in range(-n, n + 1)
            for l in range(-n, n + 1)
        ]
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
    X = np.array([1, 0, 0]) * (2 * np.pi / a)
    U = np.array([1, 1 / 4, 1 / 4]) * (2 * np.pi / a)
    K = np.array([3 / 4, 3 / 4, 0]) * (2 * np.pi / a)
    L = np.array([1 / 2, 1 / 2, 1 / 2]) * (2 * np.pi / a)

    if (choice == "U") | (choice == "u"):
        # Irreducible 1st Brillouin zone boundary path L, G, X, U, G
        k_path = np.concatenate(
            [
                np.linspace(L, Gamma, 27, endpoint=False),
                np.linspace(Gamma, X, 30, endpoint=False),
                np.linspace(X, U, 11, endpoint=False),
                np.linspace(U, Gamma, 32),
            ]
        )
    else:
        # Irreducible 1st Brillouin zone boundary path L, G, X, K, G
        k_path = np.concatenate(
            [
                np.linspace(L, Gamma, 23, endpoint=False),
                np.linspace(Gamma, X, 27, endpoint=False),
                np.linspace(X, K, 21, endpoint=False),
                np.linspace(K, Gamma, 29),
            ]
        )
    print(k_path)

    # Energy eigenvalues
    E = []
    for k in tqdm(k_path, desc="Calculating energy eigenvalues"):  # Progress bar
        e = np.linalg.eigvalsh(H_fill(k))  # Computes eigenvalues e
        E.append(e)


    # Plotting
    plt.figure(figsize=(10, 6))
    for i, band in enumerate(np.array(E).T[:7]):
        plt.plot(np.linspace(0, 1, len(k_path)), band, label=f"Band {i+1}")

    high_symmetry_points = []
    labels = []

    if (choice == "U") | (choice == "u"):
        # Add vertical lines for high-symmetry points in path L, G, X, U, G
        high_symmetry_points = [0, 27, 57, 68, 100]
        labels = ["L", "Γ", "X", "U", "Γ"]
    else:
        # Add vertical lines for high-symmetry points in path L, G, X, K, G
        high_symmetry_points = [0, 23, 50, 71, 100]
        labels = ["L", "Γ", "X", "K", "Γ"]

    for point in high_symmetry_points:
        plt.axvline(x=point / len(k_path), color="k", linestyle="--")
    plt.xticks([point / len(k_path) for point in high_symmetry_points], labels)

    plt.xlabel("k-path (Å)")
    plt.ylabel("Energy ($eV$)")
    plt.title("Band Structure of Silicon")
    plt.legend()
    plt.grid(True)
    plt.show()
