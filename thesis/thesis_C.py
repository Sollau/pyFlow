import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Importing tqdm for the progress bar
import scipy.constants as const

a = 5.431e-10  # Lattice constant of Silicon (in m)
N = 100  # Number of points in the reciprocal space

tau = (1 / 8) * np.ones(
    3
)  # Offset from central node in Bravais lattice (in units of a)

basis = np.array(
    [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
)  # Base vectors of the reciprocal lattice (in units of 2*np.pi/a)

# Conversion factors
J_to_eV = 6.2415e18
Ry_to_eV = 13.6057039763


def V_ps(G):
    """Form factors of the pseudopotential (in eV), given a vector G of the reciprocal lattice."""
    if np.dot(G, G) == 3:
        return -0.2241 * Ry_to_eV
    elif np.dot(G, G) == 8:
        return 0.0551 * Ry_to_eV
    elif np.dot(G, G) == 11:
        return 0.0724 * Ry_to_eV
    else:
        return 0


def V_ps_TOT(G):
    """Total pseudopotential (in eV), given a vector G of the reciprocal lattice."""
    S = np.cos(2 * np.pi * np.dot(G, tau))  # Structure factor (dimensionless)
    V = V_ps(G) * S  # Total pseudopotential
    return V


def T(G, k):
    """Kinetic energy (in eV), given a vector k of the irreducible 1st Brillouin zone border and a vector G of the reciprocal lattice."""
    T_const = const.hbar**2 / (2 * const.m_e)  # Kinetic constant
    T = (
        T_const * (np.linalg.norm(k + G) * (2 * np.pi / a)) ** 2 * J_to_eV
    )  # Kinetic energy
    return T


def index_generator(m, states):
    """Calculate the trplets of Miller indexes of a reciprocal lattice vector, given a index m and the number of states 'states'."""
    n = (states**3) // 2
    s = m + n
    floor = states // 2
    h = s // states**2 - floor
    k = s % states**2 // states - floor
    l = s % states - floor
    return h, k, l


def vector_matrix(states, basis):
    """Construct a matrix of reciprocal lattice vectors, given the number of states 'states' and the primitive vectors 'basis'."""
    n = states**3  # order of the matrix
    G_matrix = np.zeros(shape=(n, n), dtype=object)
    for i in range(n):
        G_matrix[i][i] += index_generator(i - n // 2, states) @ basis
        for j in range(n):
            G_matrix[i][j] += index_generator(i - j, states) @ basis
    return G_matrix


def H_fill(G_matrix, k):
    """Construct the Hamiltonian matrix, given a vector k of the irreducible 1st Brillouin zone border."""
    # Get the order of G_matrix in order to create the hamiltonian matrix with the same dimensions
    l = np.shape(G_matrix)[0]

    # Create a void matrix to fill with real numbers
    H = np.zeros((l, l), dtype=np.float64)

    # Fill the matrix
    for i in range(l):
        for j in range(l):
            
            if i == j:
                # Add the diagonal kinetic+potential terms
                H[i][i] = T(G_matrix[i][i], k)
            else:    
                # Add the off-diagonal potential terms
                H[i][j] = V_ps_TOT(G_matrix[i][j])

    return H


if __name__ == "__main__":

    # High-symmetry points in units of 2*np.pi/a
    L = np.array([1 / 2, 1 / 2, 1 / 2])
    Gamma = np.array([0, 0, 0])
    X = np.array([0, 0, 1])  # [1, 0, 0]
    U = np.array([1 / 4, 1 / 4, 1])  # [1, 1 / 4, 1 / 4]
    K = np.array([3 / 4, 3 / 4, 0])

    # Irreducible 1st Brillouin zone boundary path L, G, X, U|K, G
    k_path = np.concatenate(
        [
            np.linspace(L, Gamma, 100, endpoint=False),  # 26
            np.linspace(Gamma, X, 100, endpoint=False),  # 26
            np.linspace(X, U, 25, endpoint=False),  # 8
            np.linspace(K, Gamma, 100),  # 20
        ]
    )

    G = vector_matrix(7, basis=basis)

    # Energy eigenvalues
    E = []
    for k in tqdm(k_path, desc="Calculating energy eigenvalues"):  # Progress bar
        e = np.linalg.eigvalsh(
            H_fill(G, k)
        )  # Computes eigenvalues e (assuming H symmetrycal)
        E.append(e)

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, band in enumerate(np.array(E).T[:7]):
        plt.plot(np.linspace(0, 1, len(k_path)), band, label=f"Band {i+1}")

    # Add vertical lines for high-symmetry points in path L, G, X, U, G
    high_symmetry_points = [0, 100, 200, 225, 325]  # [0, 26, 52, 60, 100]
    labels = ["L", "Γ", "X", "U,K", "Γ"]

    for point in high_symmetry_points:
        plt.axvline(x=point / len(k_path), color="k", linestyle="--")
    plt.xticks([point / len(k_path) for point in high_symmetry_points], labels)

    plt.xlabel("k-path (Å)")
    plt.ylabel("Energy ($eV$)")
    plt.title("Band Structure of Silicon")
    plt.legend()
    plt.grid(True)

    plt.savefig("BandStructure_C.png")
    plt.show()
