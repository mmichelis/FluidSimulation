import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from matplotlib import animation


def coord (i, j):
    """
    Coordinate mapping from matrix index to 2D x y coordinate.
    i is the row entry index, j is the column entry index
    """
    return j, i


def interpolate (u, v, x):
    """
    Find nearby grid points and return a velocity vector that interpolates those grid points.
    u is the horizontal component velocity field
    v is the vertical component velocity field
    x is the position vector
    """
    if x[0] < 0:
        x[0] = 0
    if x[0] > width-2:
        x[0] = width-2
    if x[1] < 0:
        x[1] = 0
    if x[1] > height-2:
        x[1] = height-2
    
    xi = int(x[1])
    xj = int(x[0])

    ax = (x[0] - xj) / h
    ay = (x[1] - xi) / h

    # Bilinear interpolation in 2D
    uij = (1-ax)*(1-ay)*u[xi,xj] + (1-ax)*ay*u[xi+1,xj] + ax*(1-ay)*u[xi,xj+1]          + ax*ay*u[xi+1,xj+1]
    vij = (1-ax)*(1-ay)*v[xi,xj] + (1-ax)*ay*v[xi+1,xj] + ax*(1-ay)*v[xi,xj+1]          + ax*ay*v[xi+1,xj+1]

    return uij, vij


def advect (u, v):
    """
    Advect the velocity fields using Semi-Lagrangian advection.
    u is the horizontal component velocity field
    v is the vertical component velocity field
    """
    # NOTICE: memory usage might be too high, could optimize

    # Store the values from timestep n
    un = u
    vn = v

    for i in range (height):
        for j in range (width):
            oldpos = coord (i,j) - dt * np.stack((u[i,j], v[i,j]))
            u[i,j], v[i,j] = interpolate (un, vn, oldpos)


    # Return values for timestep n+1
    return u, v


def extforce (u, v):
    """
    Apply external forces on the velocity field.
    u is the horizontal component velocity field
    v is the vertical component velocity field
    """

    for i in range (height):
        for j in range (width):
            u[i,j], v[i,j] = np.stack((u[i,j], v[i,j])) + dt * extacc

    return u, v


def index (i, j):
    # how we find the index of point i,j in row of matrix A
    return i*width + j

def project (u, v):
    """
    Calculate pressure field such that velocity field is divergence free.
    u is the horizontal component velocity field
    v is the vertical component velocity field
    """

    # Construct linear system Ap = d
    A = sps.lil_matrix ((width*height, width*height))
    d = np.zeros ((width*height))

    for i in range (1, height-1):
        for j in range (1, width-1):
            A[index(i,j), index(i,j)] = 4
            A[index(i,j), index(i-1,j)] = -1
            A[index(i,j), index(i+1,j)] = -1
            A[index(i,j), index(i,j-1)] = -1
            A[index(i,j), index(i,j+1)] = -1
            
            d[index(i,j)] = -1/h * (u[i,j] - u[i,j-1] + v[i,j] - v[i-1,j])

    # Unhandled boundary cases, we assume solid walls that don't move
    A[index(0,0), index(0,0)] = 2
    A[index(0,0), index(1,0)] = -1
    A[index(0,0), index(0,1)] = -1
    d[index(0,0)] = -1/h * (u[0,0] + v[0,0])

    A[index(height-1,0), index(0,0)] = 2
    A[index(height-1,0), index(height-1,1)] = -1
    A[index(height-1,0), index(height-2,0)] = -1
    d[index(height-1,0)] = -1/h * (u[height-1,0] - v[height-2,0])

    A[index(0,width-1), index(0,width-1)] = 2
    A[index(0,width-1), index(1,width-1)] = -1
    A[index(0,width-1), index(0,width-2)] = -1
    d[index(0,width-1)] = -1/h * (-u[0,width-2] + v[0,width-1])

    A[index(height-1,width-1), index(height-1,width-1)] = 2
    A[index(height-1,width-1), index(height-2,width-1)] = -1
    A[index(height-1,width-1), index(height-1,width-2)] = -1
    d[index(height-1,width-1)] = -1/h * (-u[height-1,width-2] - v[height-2,width-1])


    for i in range (1, height-1):
        A[index(i,0), index(i,0)] = 3
        A[index(i,0), index(i-1,0)] = -1
        A[index(i,0), index(i+1,0)] = -1
        A[index(i,0), index(i,1)] = -1
        d[index(i,0)] = -1/h * (u[i,0] + v[i,0] - v[i-1,0])

    for i in range (1, height-1):
        A[index(i,width-1), index(i,width-1)] = 3
        A[index(i,width-1), index(i-1,width-1)] = -1
        A[index(i,width-1), index(i+1,width-1)] = -1
        A[index(i,width-1), index(i,width-2)] = -1
        d[index(i,width-1)] = -1/h * (- u[i,width-2] + v[i, width-1] - v[i-1,width-1])

    for j in range (1, width-1):
        A[index(0,j), index(0,j)] = 3
        A[index(0,j), index(1,j)] = -1
        A[index(0,j), index(0,j-1)] = -1
        A[index(0,j), index(0,j+1)] = -1
        d[index(0,j)] = -1/h * (u[0,j] - u[0,j-1] + v[0,j])
    
    for j in range (1, width-1):
        A[index(height-1,j), index(height-1,j)] = 3
        A[index(height-1,j), index(height-2,j)] = -1
        A[index(height-1,j), index(height-1,j-1)] = -1
        A[index(height-1,j), index(height-1,j+1)] = -1
        d[index(height-1,j)] = -1/h * (u[height-1,j] - u[height-1,j-1] - v[height-2,j])


    A = A * dt / (density * h**2)

    A = sps.csr_matrix (A)
    p = np.reshape(spsolve (A, d), (height, width))

    # Calculate new velocity field based on this pressure field
    for i in range (height):
        for j in range (width):
            if (i == height-1 and j == width-1) or (i == height-1 and j == 0) or (i == 0 and j == width-1) or (i == 0 and j == 0):
                # Set vertical velocity to movement of solid wall 0
                u[i,j] = 0
                v[i,j] = 0
            elif i == height-1 or i == 0:
                u[i,j] = u[i,j] - dt / (density * h) * (p[i,j+1] - p[i,j])
                v[i,j] = 0
            elif j == width-1 or j == 0:
                u[i,j] = 0
                v[i,j] = v[i,j] - dt / (density * h) * (p[i+1,j] - p[i,j])
            else:
                u[i,j] = u[i,j] - dt / (density * h) * (p[i,j+1] - p[i,j])
                v[i,j] = v[i,j] - dt / (density * h) * (p[i+1,j] - p[i,j])

    # let's get some inflow
    u[4:12, 0] = 1

    return u, v, p


### 2D fluid simulation with semi lagrangian advection and pressure projection
width = 20
height = 20
timesteps = 50
dt = 0.001
h = 0.01
density = 1000

SLEEPMS = 20

extacc = np.array([0, -9.81])   # External force as acceleration in 2D, here only gravity


# Initialize horizontal and vertical velocity fields and pressure field
u = np.zeros ((height, width))
v = np.zeros ((height, width))
p = np.zeros ((height, width))


# Array that stores temporal evolution of norm of velocity field
q = np.zeros ((timesteps, height, width))
qu = np.zeros ((timesteps, height, width))
qv = np.zeros ((timesteps, height, width))
qp = np.zeros ((timesteps, height, width))

q[0,:,:] = u**2 + v**2
qu[0,:,:] = u
qv[0,:,:] = v
qp[0,:,:] = p


for t in range (1, timesteps):
    ### Splitting the fluid equations

    # Advect the fields
    u, v = advect (u, v)

    # Body forces
    u, v = extforce (u, v)

    # Pressure projection
    u, v, p = project (u, v)

    #p[4:12,4:12] = 1000

    q[t,:,:] = u**2 + v**2
    qu[t,:,:] = u
    qv[t,:,:] = v
    qp[t,:,:] = p



fig = plt.figure (1, figsize=(10,10))
ax = fig.add_subplot (221)
axp = fig.add_subplot (222)
axu = fig.add_subplot (223)
axv = fig.add_subplot (224)

#im = ax.quiver (qu[0,:,:], qv[0,:,:])
title = ax.text(0.5, 0.97, "Current time: 0", bbox={'facecolor':'w', 'alpha':0.5, 'pad':2}, transform=ax.transAxes, ha="center")

im = ax.imshow (q[0,:,:])
imp = axp.imshow (qp[0,:,:])
imu = axu.imshow (qu[0,:,:])
imv = axv.imshow (qv[0,:,:])
plt.colorbar (im)
plt.colorbar (imp)
plt.colorbar (imu)
plt.colorbar (imv)
im.axes.figure.canvas.draw()
imp.axes.figure.canvas.draw()
imu.axes.figure.canvas.draw()
imv.axes.figure.canvas.draw()


def animate (t):
    title.set_text("Current time: %.5f" % (t*dt))
    #im.set_UVC (qu[t,:,:], qv[t,:,:])
    im.set_data (q[t,:,:])
    imp.set_data (qp[t,:,:])
    imu.set_data (qu[t,:,:])
    imv.set_data (qv[t,:,:])

    return im


anim = animation.FuncAnimation (fig, animate, frames=timesteps, interval=SLEEPMS, blit=False, repeat=True)

anim.save('Fluid.mp4', fps=30)

plt.show ()
    



