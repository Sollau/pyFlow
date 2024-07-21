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

def ComplementaryMinor(B, x, y):
    "Calculates the complementary minor of a matrix, given the indexes of row x and column y."
    Bminor=[]
    for h in range (0, len(B)):
        Bminor.append([])
        for i in range (0,len(B)):
            Bminor[h].append(B[h][i])
    for j in range (0, len(B)):
        Bminor[j] = Bminor[j][:y]+Bminor[j][y+1:] 
    Bminor.remove(Bminor[x])
    return Bminor

def laplace(C):
    "Calculates the determinant of a NxN matrix, using the Lapalace method."
    det = 0
    if len(C)==1:
        det+=C[0][0]    
    elif len(C)==2:
        det+=det2x2(C)
        print("This is 2x2det: " + str(det2x2(C)))
        print(C)
    else:    
        for i in range(0, len(C)):
            det+=(-1)**i*C[0][i]*laplace(ComplementaryMinor(C, 0, i))
    
    return det

print(laplace(A))