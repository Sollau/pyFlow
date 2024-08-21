import numpy as np
from scipy import constants as c
from matplotlib import pyplot as plt
from tqdm import tqdm  # Importing tqdm for the progress bar

# Constants
a = 5.43e-10  # Lattice constant in meters
kinetic_constant = c.hbar**2 / (
    2 * c.m_e * c.e
)  # Kinetic energy constant in [J*s]**2/(kg*C)
n = 100  # Number of points for plotting
ry_to_ev = 13.6059  # Conversion factor from Rydberg to electronvolt

# Form factors in eV
form_factors = {
    3.0: (ry_to_ev * -0.21),
    8.0: (ry_to_ev * 0.04),
    11.0: (ry_to_ev * 0.08),
}

# Reciprocal basis in units of 2 pi / a
reciprocal_basis = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])

# Symmetry points in the Brillouin zone
g = np.array([0, 0, 0])  # Gamma point
l = np.array([1 / 2, 1 / 2, 1 / 2])  # L point
k = np.array([3 / 4, 3 / 4, 0])  # K point
x = np.array([0, 0, 1])  # X point
w = np.array([1, 1 / 2, 0])  # W point
u = np.array([1 / 4, 1 / 4, 1])  # U point


# Function to create a linear path between two points in k-space
def linpath(a, b, n=50, endpoint=True):
    spacings = [
        np.linspace(start, end, num=n, endpoint=endpoint) for start, end in zip(a, b)
    ]
    return np.stack(spacings, axis=-1)


# Define k-paths
Lambda = linpath(l, g, n, endpoint=False)
Delta = linpath(g, x, n, endpoint=False)
x_uk = linpath(x, u, n // 4, endpoint=False)
Sigma = linpath(k, g, n, endpoint=True)


# Calculate the kinetic energy term
def kinetic(k, G):
    v = k + G
    return kinetic_constant * v @ v


# Calculate the potential energy term
def potential(G, tau, sym):
    return sym * np.cos(
        2 * np.pi * G @ tau
    )  # 2*np.pi keeps into account the unit of measurements of the result of the dot product


# Construct the Hamiltonian matrix
def hamiltonian(lattice_constant, form_factors, reciprocal_basis, k, states):
    a = lattice_constant
    ff = form_factors
    basis = reciprocal_basis
    kinetic_UoM = (2 * np.pi / a) ** 2  # Unit of Measurement of the kinetic energy
    offset = 1 / 8 * np.ones(3)
    n = (
        states**3
    )  # Order of the matrix and starting point for creating a base of reciprocal lattice vectors G

    # Calculate the coefficients of the reciprocal lattice vectors G that are to be used in the hamiltonian
    def coefficients(m):
        n = (states**3) // 2
        s = m + n
        floor = states // 2
        h = s // states**2 - floor
        k = s % states**2 // states - floor
        l = s % states - floor
        return h, k, l

    # Create and then fill the hamiltonian matrix
    h = np.empty(shape=(n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                # kinetic diagonal terms
                g = coefficients(i - n // 2) @ basis
                h[i][j] = kinetic_UoM * kinetic(k, g)
            else:
                # potential terms for every couple of index i,j
                g = coefficients(i - j) @ basis
                factors = ff.get(g @ g)
                h[i][j] = potential(g, offset, factors) if factors else 0

    return h


# Calculate the band structure along a given path in k-space
def band_structure(lattice_constant, form_factors, reciprocal_basis, states, path):
    bands = []
    # Progress bar
    for k in tqdm(np.vstack(path), desc="Calculating Eigenvalues"):
        h = hamiltonian(lattice_constant, form_factors, reciprocal_basis, k, states)
        eigvals = np.linalg.eigvals(h)
        eigvals.sort()
        bands.append(eigvals[:8])
    return np.stack(bands, axis=-1)


# Execution
bands = band_structure(
    a, form_factors, reciprocal_basis, states=7, path=[Lambda, Delta, x_uk, Sigma]
)
bands -= max(bands[3])

# Create figure for plotting
plt.figure(figsize=(15, 9))

# Limits on x and y axis
plt.xlim(0, len(bands))
plt.ylim(min(bands[0]) - 1, max(bands[7]) + 1)

# Labels for k-path points along the x axis
xticks = n * np.array([0, 0.5, 1, 1.5, 2, 2.25, 2.75, 3.25])
plt.xticks(
    xticks,
    [
        "$\mathbf{L}$",
        "$\Lambda$",
        "$\mathbf{\Gamma}$",
        "$\Delta$",
        "$\mathbf{X}$",
        "$\mathbf{U,K}$",
        "$\Sigma$",
        "$\mathbf{\Gamma}$",
    ],
)

# Axis labels and graph title
plt.xlabel("k-Path")
plt.ylabel("E(k) (eV)")
plt.title("Band structure of Silicon")

# Enable grid
plt.grid(True)

# Draw bands
for band in bands:
    plt.plot(band)

# Save and show figure
plt.savefig("BandStructure_G.png")
plt.show()
