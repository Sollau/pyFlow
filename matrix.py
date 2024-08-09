import numpy as np

list_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
list_2 = [10, 20, 30, 40, 50,60, 70, 80, 90]

l = len(list_1)

M = np.zeros((l,l))

for i in range(0, l):
    M[i][i] += list_1[i]
    for j in range (0, l):
        M[i][j]+=list_1[-i]+list_2[j]

print(M)
