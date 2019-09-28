import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import time
from matplotlib import animation


SLEEPTIME = 0.001
SLEEPMS = 10

# 2D simulation, m * n grid, indexing is u(i*n + j)
m = 20
n = 30
timesteps = 200  # amount of timesteps

visc = 1    # viscosity
h = 0.1       # homogenous step size
dt = 0.001   # time step size
c = 0.01     # damping of wall bounce

# u has ux, uy, p components, u[3*(i*n + j)] is uxi, u[3*(i*n + j) + 1] is uyi, u[3*(i*n + j) + 2] is p
#u = np.zeros (3*m*n)
A = sps.lil_matrix ((3*m*n, 3*m*n))
f = np.zeros (3*m*n)

u = np.zeros ((timesteps, m, n))
v = np.zeros ((timesteps, m, n))
p = np.zeros ((m, n))


def index (i, j):
    # how we find the index of point i,j
    return 3*(i*n+j)


for i in range(1, m-1):
        for j in range(1, n-1):
                ### first equation ux
                # ux component
                A[index(i,j), index(i,j)] = 4 * visc/h**2
                A[index(i,j), index(i-1,j)] = -visc/h**2
                A[index(i,j), index(i+1,j)] = -visc/h**2
                A[index(i,j), index(i,j-1)] = -visc/h**2
                A[index(i,j), index(i,j+1)] = -visc/h**2
                # p component
                A[index(i,j), index(i,j)+2] = -1/h
                A[index(i,j), index(i,j+1)+2] = 1/h
                # right hand side
                f[index(i,j)] = 0   # for now just keep f at zeros


                ### second equation for uy
                # uy component
                A[index(i,j)+1, index(i,j)+1] = 4 * visc/h**2
                A[index(i,j)+1, index(i-1,j)+1] = -visc/h**2
                A[index(i,j)+1, index(i+1,j)+1] = -visc/h**2
                A[index(i,j)+1, index(i,j-1)+1] = -visc/h**2
                A[index(i,j)+1, index(i,j+1)+1] = -visc/h**2
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


A = sps.csr_matrix(A)
x = spsolve (A, f)



# Ufield = np.zeros((m, n))
# Vfield = np.zeros((m, n))
# Pfield = np.zeros((m, n))

for i in range(m):
    for j in range(n):
        u[0,i,j] = x[index(i,j)]
        v[0,i,j] = x[index(i,j)+1]
        p[i,j] = x[index(i,j)+2]



for t in range(1, timesteps):
    for i in range(1,m-1):
        for j in range(1,n-1):
            u[t,i,j] = u[t-1,i,j] + dt * (-1/h * (p[i,j+1] - p[i,j]) + visc/h**2 * (u[t-1,i+1,j] + u[t-1,i-1,j] + u[t-1,i,j+1] + u[t-1,i,j-1] - 4 * u[t-1,i,j]))

            v[t,i,j] = v[t-1,i,j] + dt * (-1/h * (p[i+1,j] - p[i,j]) + visc/h**2 * (v[t-1,i+1,j] + v[t-1,i-1,j] + v[t-1,i,j+1] + v[t-1,i,j-1] - 4 * v[t-1,i,j]))

    for i in range(m):
        # No energy lost at walls, just bounce
        # let's add spring constant
        u[t,i,0] = u[t-1,i,0] - c * u[t-1,i,1]
        u[t,i,n-1] = u[t-1,i,n-1] - c * u[t-1,i,n-2]

    for j in range(n):
        # No energy lost at walls, just bounce
        v[t,0,j] = v[t-1,0,j] - c * v[t-1,1,j]
        v[t,m-1,j] = v[t-1,m-1,j] - c * v[t-1,m-2,j]



fig = plt.figure (1, figsize=(10,5))
ax = fig.add_subplot (111)

#im = ax.quiver (u[0,:,:], v[0,:,:])
title = ax.text(0.5, 0.97, "Current time: 0", bbox={'facecolor':'w', 'alpha':0.5, 'pad':2}, transform=ax.transAxes, ha="center")

im = ax.imshow (np.sqrt(u[0,:,:]**2 + v[0,:,:]**2))
plt.colorbar (im)
im.axes.figure.canvas.draw()

# for t in range(timesteps):
#     ax.set_title("Current time: %.2f" % (t*dt))
#     #im.set_UVC (u[t,:,:], v[t,:,:])
#     im.set_data (np.sqrt(u[t,:,:]**2 + v[t,:,:]**2))
#     im.axes.figure.canvas.draw()
#     time.sleep (SLEEPTIME)


def init ():
    title.set_text("Current time: 0")
    im.set_data (np.sqrt(u[0,:,:]**2 + v[0,:,:]**2))

    return im, ax, title

def animate (t):
    title.set_text("Current time: %.2f" % (t*dt))
    #im.set_UVC (u[t,:,:], v[t,:,:])
    im.set_data (np.sqrt(u[t,:,:]**2 + v[t,:,:]**2))

    return im, ax, title


anim = animation.FuncAnimation (fig, animate, init_func=init, frames=timesteps, interval=SLEEPMS, blit=False, repeat=False)

anim.save('basic_animation.mp4', fps=30)

plt.show ()


