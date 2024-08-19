import numpy as np
from scipy import constants as c
from matplotlib import pyplot as plt
from tqdm import tqdm  # Importing tqdm for the progress bar

# Constants
a = 5.43e-10  # Lattice constant in meters
kinetic_constant = c.hbar**2 / (2 * c.m_e * c.e)  # Kinetic energy constant
n = 100  # Number of points for plotting
ry_to_ev = 13.6059  # Conversion factor from Rydberg to eV

# Form factors in eV
form_factors = {3.0: (ry_to_ev * -0.21), 8.0: (ry_to_ev * 0.04), 11.0: (ry_to_ev * 0.08)}

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
    spacings = [np.linspace(start, end, num=n, endpoint=endpoint) for start, end in zip(a, b)]
    return np.stack(spacings, axis=-1)

# Define k-paths
lambd = linpath(l, g, n, endpoint=False)
delta = linpath(g, x, n, endpoint=False)
x_uk = linpath(x, u, n // 4, endpoint=False)
sigma = linpath(k, g, n, endpoint=True)

# Function to calculate the coefficients for the Hamiltonian matrix
def coefficients(m, states):
    n = (states**3) // 2
    s = m + n
    floor = states // 2
    h = s // states**2 - floor
    k = s % states**2 // states - floor
    l = s % states - floor
    return h, k, l

# Function to calculate the kinetic energy term
def kinetic(k, g):
    v = k + g
    return kinetic_constant * v @ v

# Function to calculate the potential energy term
def potential(g, tau, sym):
    return sym * np.cos(2 * np.pi * g @ tau)

# Function to construct the Hamiltonian matrix
def hamiltonian(lattice_constant, form_factors, reciprocal_basis, k, states):
    a = lattice_constant
    ff = form_factors
    basis = reciprocal_basis
    kinetic_c = (2 * np.pi / a) ** 2
    offset = 1 / 8 * np.ones(3)
    n = states**3

    def coefficients(m):
        n = (states**3) // 2
        s = m + n
        floor = states // 2
        h = s // states**2 - floor
        k = s % states**2 // states - floor
        l = s % states - floor
        return h, k, l

    h = np.empty(shape=(n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                g = coefficients(i - n // 2) @ basis
                h[i][j] = kinetic_c * kinetic(k, g)
            else:
                g = coefficients(i - j) @ basis
                factors = ff.get(g @ g)
                h[i][j] = potential(g, offset, factors) if factors else 0

    return h

# Function to calculate the band structure along a given path in k-space
def band_structure(lattice_constant, form_factors, reciprocal_basis, states, path):
    bands = []
    for k in tqdm(np.vstack(path), desc="Calculating Eigenvalues"):  # Adding progress bar
        h = hamiltonian(lattice_constant, form_factors, reciprocal_basis, k, states)
        eigvals = np.linalg.eigvals(h)
        eigvals.sort()
        bands.append(eigvals[:8])
    return np.stack(bands, axis=-1)

# Calculate band structure
bands = band_structure(a, form_factors, reciprocal_basis, states=7, path=[lambd, delta, x_uk, sigma])
bands -= max(bands[3])

# Plotting the band structure
plt.figure(figsize=(15, 9))
plt.xlim(0, len(bands))
plt.ylim(min(bands[0]) - 1, max(bands[7]) + 1)
xticks = n * np.array([0, 0.5, 1, 1.5, 2, 2.25, 2.75, 3.25])
plt.xticks(
    xticks,
    ["$L$", "$\Lambda$", "$\Gamma$", "$\Delta$", "$X$", "$U,K$", "$\Sigma$", "$\Gamma$"],
)
plt.xlabel("k-Path")
plt.ylabel("E(k) (eV)")
plt.grid(True)
for band in bands:
    plt.plot(band)
plt.show()
