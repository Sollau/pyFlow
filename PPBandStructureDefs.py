def coefficients(m, states):
    """Coefficients (h, k, l) for vectors in the reciprocal lattice base."""    
    n = (states**3) // 2
    s = m + n
    floor = states // 2

    h = s // states**2 - floor
    k = s % states**2 // states - floor
    l = s % states - floor

    return h, k, l



from scipy import constants as c

KINETIC_CONSTANT = c.hbar**2 / (2 * c.m_e * c.e)

def kinetic(k, g):
    """Remove constant from function definition so it is not recalculated every time"."""    
    v = k + g
    
    return KINETIC_CONSTANT * v @ v



import numpy as np

def potential(g, tau, sym, asym=0): #asym=0 for silicon
    """Potential component."""
    return sym * np.cos(2 * np.pi * g @ tau) # + 1j * asym * np.sin(2 * np.pi * g @ tau)



import functools
import itertools

def hamiltonian(lattice_constant, form_factors, reciprocal_basis, k, states):
    """Hamiltonian of the system."""
    # renaming variables for faster coding
    a = lattice_constant
    ff = form_factors
    basis = reciprocal_basis
    
    # some constants that don't need to be recalculated
    kinetic_c = (2 * np.pi / a)**2
    offset = 1 / 8 * np.ones(3)
    
    # states determines size of matrix
    # each of the three reciprocal lattice vectors can
    # take on this many states, centered around zero
    # resulting in an n**3 x n**3 matrix
    
    n = states**3
    
    # internal cached implementation
    @functools.lru_cache(maxsize=n)
    def coefficients(m):
        n = (states**3) // 2
        s = m + n
        floor = states // 2

        h = s // states**2 - floor
        k = s % states**2 // states - floor
        l = s % states - floor

        return h, k, l
    
    # initialize our matrix to arbitrary elements
    # from whatever's currently in the memory location
    # these will be filled up anyway and it's faster than initializing to a value
    h = np.empty(shape=(n, n))
    
    # cartesian product over rows, cols; think an odometer
    for row, col in itertools.product(range(n), repeat=2):
        
        if row == col:
            g = coefficients(row - n // 2) @ basis
            h[row][col] = kinetic_c * kinetic(k, g)
            
        else:
            g = coefficients(row - col) @ basis
            factors = ff.get(g @ g)
            # potential is 0 for g**2 != (3, 8, 11)
            h[row][col] = potential(g, offset, *factors) if factors else 0
    
    return h



def band_structure(lattice_constant, form_factors, reciprocal_basis, states, path):
    """Calculate the first eight eigenvalues of the Hamiltonian for each point along the k-path."""
    bands = []
    
    # vstack concatenates our list of paths into one nice array
    for k in np.vstack(path):
        h = hamiltonian(lattice_constant, form_factors, reciprocal_basis, k, states)
        eigvals = np.linalg.eigvals(h)
        eigvals.sort()
        # picks out the lowest eight eigenvalues
        bands.append(eigvals[:8])
    
    
    return np.stack(bands, axis=-1)




def linpath(a, b, n=50, endpoint=True):
    """
    Create an array of n equally spaced points along the path a -> b, inclusive.

    args:
        a: An iterable of numbers that represents the starting position.
        b: An iterable of numbers that represents the ending position.
        n: The integer number of sample points to calculate. Defaults to 50.
        
    returns:
        A numpy array of shape (n, k) where k is the shortest length of either
        iterable -- a or b.
    """
    # list of n linear spacings between the start and end of each corresponding point
    spacings = [np.linspace(start, end, num=n, endpoint=endpoint) for start, end in zip(a, b)]
    
    # stacks along their last axis, transforming a list of spacings into an array of points of len n
    return np.stack(spacings, axis=-1)