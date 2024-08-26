import numpy as np

a = np.array([1, 2, 3])
b = np.array([6, 2, 5])

basis = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])

G00 = np.array([0, -1, -1])
G01 = np.array([0, 0, -1])

c = np.pi / 2

print(a, b, c)

print(c * a)
print(b * c)

print(a + b)
print(a @ b)

print(np.linalg.norm(a))

print(G00 @ basis)
print(G01 @ basis)
