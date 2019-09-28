import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve


# 2D simulation, m * n grid, indexing is u(i*n + j)
m = 20
n = 30

v = 1   # viscosity
h = 1   # homogenous step size

# u has ux, uy, p components, u[3*(i*n + j)] is uxi, u[3*(i*n + j) + 1] is uyi, u[3*(i*n + j) + 2] is p
#u = np.zeros (3*m*n)
A = sps.lil_matrix ((3*m*n, 3*m*n))
f = np.zeros (3*m*n)


def index (i, j):
    # how we find the index of point i,j
    return 3*(i*n+j)


for i in range(1, m-1):
        for j in range(1, n-1):
                ### first equation ux
                # ux component
                A[index(i,j), index(i,j)] = 4 * v/h**2
                A[index(i,j), index(i-1,j)] = -v/h**2
                A[index(i,j), index(i+1,j)] = -v/h**2
                A[index(i,j), index(i,j-1)] = -v/h**2
                A[index(i,j), index(i,j+1)] = -v/h**2
                # p component
                A[index(i,j), index(i,j)+2] = -1/h
                A[index(i,j), index(i,j+1)+2] = 1/h
                # right hand side
                f[index(i,j)] = 0   # for now just keep f at zeros


                ### second equation for uy
                # uy component
                A[index(i,j)+1, index(i,j)+1] = 4 * v/h**2
                A[index(i,j)+1, index(i-1,j)+1] = -v/h**2
                A[index(i,j)+1, index(i+1,j)+1] = -v/h**2
                A[index(i,j)+1, index(i,j-1)+1] = -v/h**2
                A[index(i,j)+1, index(i,j+1)+1] = -v/h**2
                # p component
                A[index(i,j)+1, index(i,j)+2] = -1/h
                A[index(i,j)+1, index(i+1,j)+2] = 1/h
                # right hand side
                f[index(i,j)+1] = 0   # for now just keep f at zeros


                ### third equation for conservation
                # ux component
                A[index(i,j)+2, index(i,j)] = 1/h
                A[index(i,j)+2, index(i,j-1)] = -1/h
                # uy component
                A[index(i,j)+2, index(i,j)+1] = 1/h
                A[index(i,j)+2, index(i-1,j)+1] = -1/h
                # right hand side
                f[index(i,j)+2] = 0


# Boundary Conditions
# eh too lazy, put everything to zero
# upper and lower grid rows
for j in range(n):
        A[index(0,j), index(0,j)] = 1
        A[index(0,j)+1, index(0,j)+1] = 1
        A[index(0,j)+2, index(0,j)+2] = 1

        f[index(0,j)] = 10
        f[index(0,j)+1] = 0
        f[index(0,j)+2] = 0

        A[index(m-1,j), index(m-1,j)] = 1
        A[index(m-1,j)+1, index(m-1,j)+1] = 1
        A[index(m-1,j)+2, index(m-1,j)+2] = 1

        f[index(m-1,j)] = 10
        f[index(m-1,j)+1] = 0
        f[index(m-1,j)+2] = 0

# leftest and rightest grid rows
for i in range(m):
        A[index(i,0), index(i,0)] = 1
        A[index(i,0)+1, index(i,0)+1] = 1
        A[index(i,0)+2, index(i,0)+2] = 1

        f[index(i,0)] = 10
        f[index(i,0)+1] = 0
        f[index(i,0)+2] = 10

        A[index(i,n-1), index(i,n-1)] = 1
        A[index(i,n-1)+1, index(i,n-1)+1] = 1
        A[index(i,n-1)+2, index(i,n-1)+2] = 1

        f[index(i,n-1)] = 0
        f[index(i,n-1)+1] = 0
        f[index(i,n-1)+2] = 0

# Penalty method for obstacle in field (Dirichlet BC)
beta = 1e10

for i in range(5, 15):
        for j in range(10, 15):
                A[index(i,j), index(i,j)] += beta
                A[index(i,j)+1, index(i,j)+1] += beta

                f[index(i,j)] += beta*0
                f[index(i,j)+1] += beta*0



#print ("Our A matrix: ", A)
#print ("Our f vector: ", f)


A = sps.csr_matrix(A)
u = spsolve (A, f)

#print (u)


Ufield = np.zeros((m, n))
Vfield = np.zeros((m, n))
Pfield = np.zeros((m, n))

for i in range(m):
    for j in range(n):
        Ufield[i,j] = u[index(i,j)]
        Vfield[i,j] = u[index(i,j)+1]
        Pfield[i,j] = u[index(i,j)+2]


#plt.quiver(Ufield, Vfield)
U = np.sqrt(Ufield**2 + Vfield**2)

plt.figure (1, figsize = (10,5))
plt.subplot (121)
plt.imshow(U)
plt.title('Velocity field')

plt.subplot (122)
plt.imshow(Pfield)
plt.title('Pressure field')

plt.colorbar()
plt.show()

