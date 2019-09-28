import numpy as np
import matplotlib.pyplot as plt
import time


# Solve the diffusion equation du/dt = v d2u/dx2

# Initial conditions
n = 20  # width of the simulation box
m = 10  # height of the box
timesteps = 100  # number of timesteps
dt = 0.01
dx = 0.1
dy = 0.1

visc = 0.05

u = np.zeros ((m,n))  # x direction velocities
v = np.zeros ((m,n))  # y direction velocities

u[1:-1,9] = 0.1
v[5, 1:-1] = 0.1

# Boundary Conditions : Entry
#u[1:-1,0] = 0.5


# Drawing the fluid
fig = plt.figure (1, figsize = (16,8))
ax = fig.add_subplot (111)
ax.set_title("Fluid Diffusion")

#im = ax.quiver(u, v, scale=1)
im = ax.imshow (u+v)
fig.show()
im.axes.figure.canvas.draw()


for t in range (timesteps):
    un = u  # u at the previous timestep n
    vn = v

    for i in range (1, m-1):
        for j in range (1, n-1):
            u[i,j] = un[i,j]  + visc * dt * ((un[i,j-1] - 2*un[i,j] + un[i, j+1])/dx**2 + (un[i-1,j] - 2*un[i,j] + un[i+1, j])/dy**2)

            v[i,j] = vn[i,j]  + visc * dt * ((vn[i,j-1] - 2*vn[i,j] + vn[i, j+1])/dx**2 + (vn[i-1,j] - 2*vn[i,j] + vn[i+1, j])/dy**2)

    ax.set_title("Current time: " + str(t*dt))
    #im.set_UVC(u, v)
    im.set_data(u+v)
    im.axes.figure.canvas.draw()
    time.sleep(0.1)

plt.show()