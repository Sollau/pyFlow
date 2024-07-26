"""Calculates the band structure for a 1D chain of monoelectronic atoms."""

import numpy as np
import matplotlib.pyplot as plt

a = 10 #passo del reticolo
    
m = 0.511E6 #electron mass (eV)
e = 1.062E-19 #electron charge (C)
hbar = 6.582E-16 #reduced plank constant (eV*s)

def laplacianOfExpo(f, k):
    """This is the laplacian of f = exp(i*k*x)."""
    return -(k**2) * f

n = 1000 #number of divisions
x = np.linspace(1, a, n) #creates equispaced points of x axis

for k in range(0, 5):
    phi = np.exp(1j * k * x) #expression of phi in real space
    T = hbar**2 / (2 * m) * laplacianOfExpo(phi, k) #kinetic energy
    H = T - ( e**2/ x) #hamilonian
    plt.plot(x, np.real(H), label="k = " + str(k)) #plots Energy versus Space curves !!doubts on energy handling!!  

#graph aesthetics
plt.legend()
plt.xlabel("x[a]")
plt.ylabel("H[eV]")
#plt.ylim(0,)

plt.show() #display graph
