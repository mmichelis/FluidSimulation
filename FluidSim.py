import numpy as np
import matplotlib.pyplot as plt
import time



# Initial conditions
n = 20  # width of the simulation box
m = 20  # height of the box
timesteps = 1000  # number of timesteps
dt = 0.2
dx = 1
dy = 1

visc = 0.5

u = np.zeros ((m,n))  # x direction velocities
v = np.zeros ((m,n))  # y direction velocities

u[3:-3,9] = 2
v[5, 8:-8] = 2

# Boundary Conditions : Entry
u[1:-1,0] = 1


# Drawing the fluid
fig = plt.figure (1, figsize = (16,8))
ax = fig.add_subplot (111)

#im = ax.quiver(u, v, scale=10)
im = ax.imshow (u+v)
fig.show()
im.axes.figure.canvas.draw()

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

    x = int(event.xdata+0.5)
    y = int(event.ydata+0.5)
    u[y,x] = 1
    v[y,x] = 1

cid = fig.canvas.mpl_connect('button_press_event', onclick)


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

            fuv = visc * dt * ((un[i,j-1] - 2*un[i,j] + un[i, j+1])/dx**2 + (un[i-1,j] - 2*un[i,j] + un[i+1, j])/dy**2)

            u[i,j] = un[i,j] + du + fuv


            # same for velocity in y direction
            dvx = dt * un[i,j]*(vn[i,j+1] - vn[i,j])/(dx)
            dvpx = -dt * un[i,j-1]*(vn[i,j] - vn[i,j-1])/(dx)

            dvy = dt * vn[i,j]*(vn[i+1,j] - vn[i,j])/(dy)
            dvpy = -dt * vn[i-1,j]*(vn[i,j] - vn[i-1,j])/(dy)

            dv = dvpx + dvx + dvpy + dvy

            fvv = visc * dt * ((vn[i,j-1] - 2*vn[i,j] + vn[i, j+1])/dx**2 + (vn[i-1,j] - 2*vn[i,j] + vn[i+1, j])/dy**2)

            v[i,j] = vn[i,j] + dv + fvv


    ax.set_title("Current time: " + str(t*dt))
    #im.set_UVC(u, v)
    im.set_data(u+v)
    im.axes.figure.canvas.draw()
    #time.sleep(0.1)
    plt.waitforbuttonpress (timeout=0.01)

plt.show()