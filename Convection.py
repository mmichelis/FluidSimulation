import numpy as np
import matplotlib.pyplot as plt
import time



# Initial conditions
n = 20  # width of the simulation box
m = 10  # height of the box
timesteps = 100  # number of timesteps
dt = 0.1
dx = 1
dy = 1

visc = 0.1

u = np.zeros ((m,n))  # x direction velocities
v = np.zeros ((m,n))  # y direction velocities

u[3:-3,9] = 1
v[5, 8:-8] = 1

# Boundary Conditions : Entry
u[1:-1,0] = 3


# Drawing the fluid
fig = plt.figure (1, figsize = (14,7))
ax = fig.add_subplot (111)

#im = ax.quiver(u, v, scale=1)
im = ax.imshow (u+v)
fig.show()
im.axes.figure.canvas.draw()


for t in range (timesteps):
    un = u  # u at the previous timestep n
    vn = v

    for i in range (1,m-1):
        for j in range (1,n-1):
            # negative if du goes out of volume, positive if it goes in
            dux = dt * un[i,j]*(un[i,j+1] - un[i,j])/(dx)
            dupx = -dt * un[i,j-1]*(un[i,j] - un[i,j-1])/(dx)   # previous coming into the current volume

            duy = dt * vn[i,j]*(un[i+1,j] - un[i,j])/(dy)
            dupy = -dt * vn[i-1,j]*(un[i,j] - un[i-1,j])/(dy)

            du = dupx + dux + dupy + duy

            u[i,j] = un[i,j] + du


            # same for velocity in y direction
            dvx = dt * un[i,j]*(vn[i,j+1] - vn[i,j])/(dx)
            dvpx = -dt * un[i,j-1]*(vn[i,j] - vn[i,j-1])/(dx)

            dvy = dt * vn[i,j]*(vn[i+1,j] - vn[i,j])/(dy)
            dvpy = -dt * vn[i-1,j]*(vn[i,j] - vn[i-1,j])/(dy)

            dv = dvpx + dvx + dvpy + dvy

            v[i,j] = vn[i,j] + dv

    ax.set_title("Current time: " + str(t*dt))
    #im.set_UVC(u, v)
    im.set_data(u+v)
    im.axes.figure.canvas.draw()
    time.sleep(0.01)

plt.show()