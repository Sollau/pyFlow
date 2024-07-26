import matplotlib.pyplot as plt

#parameters ?
m = 0.5 
e = 1.06*10**-19
p = 1
Z = 14 
r = 1**-10

#kinetic energy
k = p**2/(2*m)

#potential energy term (coulomb)
U = -Z*e**2/r

#hamiltonian
H = k + U

#schroedinger eq
