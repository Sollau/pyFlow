"""Calculates the band structure for a 1D chain of monoelectronic atoms."""

import numpy as np
import matplotlib.pyplot as plt

a = 10 #passo del reticolo
    
m = 0.511E6 #massa elettrone (eV)
e = 1.062E-19 #carica elettrone (C)
hbar = 6.582E-16 #costante di plank ridotta (eV*s)


def laplacianOfExpo(f):
    """This is the laplacian of f = exp(i*k*x)."""
    return -(k**2)*f

n=30

x = np.linspace(1, a, n) 

for k in range (1,8):
    phi = np.exp((0+1j)*k*x)
    T = hbar**2/(2*m)*laplacianOfExpo(phi)
    H =T-(1/x+k)
    plt.plot(x, H, label = "k = "+str(k))

plt.legend()
plt.xlabel("x[a]")
plt.ylabel("H[eV]")
plt.show()