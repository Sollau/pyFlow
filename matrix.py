A = []
a_row =[]
N = int(input("Insert the size N of your NxN matrix: "))

#create matrix element by element
for i in range (0, N):
    for j in range (0, N):
        a_row.append(int(input("a_" + str(i+1) + "," + str(j+1) + "= ")))
    A.append(a_row)
    a_row=[]

#display matrix
for k in range (0,N):
    print(A[k])

def det2x2(M):
    "Calculates the determinant of a 2x2 matrix."
    return (M[0][0]*M[1][1])-(M[0][1]*M[1][0])


def principalMinor(B, x, y):
    "Calculates the principal minor of a matrix, given the indexes of row x and column y."
    Bminor=B
    for i in range (0, len(B)):
        Bminor[i].remove(Bminor[i][y-1])
    Bminor.remove(Bminor[x-1])
    return Bminor

def laplace(C):
    "Calculates the determinant of a NxN matrix, using the Lapalace method."
    det = 0
    Cminors = C
    while int(len(Cminors))>2:
        for i in range (0, len(Cminors)):
            det+= (-1)**i*C[0][i]*laplace(principalMinor(Cminors, 1, i+1))
            Cminors = principalMinor(C, 1, i+1)
    det+=det2x2(Cminors)
    return det

print(laplace(A))    
    
    