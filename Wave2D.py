import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.cm as cm

# 2D wave equation d2h/dt2 = c1 * (d2h/dx2 + d2h/dy2), but we take dx = dy
m = 50
n = 50
timesteps = 700

dt = 1
dx = 1
c1 = 0.1

save = 1

uplim = 3
lowlim = -3

t1 = 1
t2 = 1

damp = 0.3

# special constant c = c1 * dt^2 / dx^2
c = c1 * dt**2 / dx**2

h = np.zeros ((timesteps, m, n))

# Initial conditions
width = 15
offset = 0
r = np.linspace (0, np.pi, width)
#h[1, offset:offset+width, 1] = np.sin (r)
#h[1, 1, offset:offset+width] = np.sin (r)


def animate (t, h, p):
    # Sine left bound
    #h[t,:,0] = np.sin (t/10) * np.ones (m)

    if (t > 1):
        #At all the boundaries
        h[t,0,:] = damp*h[t-1,1,:]
        h[t,n-1,:] = damp*h[t-1,n-2,:]
        h[t,:,0] = damp*h[t-1,:,1]
        h[t,:,n-1] = damp*h[t-1,:,n-2]

        # h[t,0] = c * (h[t-1,0] + h[t-1,2] - 2*h[t-1,1]) \
        #            + 2*h[t-1,0] - h[t-2,0] 

        # h[t,n-1] = c * (h[t-1,n-3] + h[t-1,n-1] - 2*h[t-1,n-2]) \
        #                + 2*h[t-1,n-1] - h[t-2,n-1] 

        if (t < 10):
            h[t,0:2,0:2] = 3 * np.ones ((2,2))

        for i in range (1,m-1):
            for j in range (1,n-1):
                # In the middle
                h[t,i,j] = (c * (h[t-1,i-1,j] + h[t-1,i+1,j] - 2*h[t-1,i,j] \
                              + h[t-1,i,j-1] + h[t-1,i,j+1] - 2*h[t-1,i,j]) \
                              + 2*t1*h[t-1,i,j] - t2*h[t-2,i,j]) * damp

    #title.set_text("Current time: %.2f" % (t*dt))
    p[0].remove ()
    p[0] = ax.plot_surface (x, y, h[t,:,:], cmap="magma")
    return p


fig = plt.figure (figsize=(10,8))
ax = fig.add_subplot (111, projection = '3d', xlim = (0, m), ylim = (0, n), zlim = (lowlim, uplim))
x = range (0, m, dx)
y = range (0, n, dx)
x, y = np.meshgrid (x,y)
p = [ax.plot_surface (x, y, h[0,:,:], color='0.75', rstride=1, cstride=1)]


ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

mm = cm.ScalarMappable(cmap=cm.jet)
mm.set_array(h)
plt.colorbar(mm)

#title = ax.text(0.5, 0.97, "Current time: 0", bbox={'facecolor':'w', 'alpha':0.5, 'pad':2}, transform=ax.transAxes, ha="center")


anim = animation.FuncAnimation (fig, animate, frames=timesteps, interval=1, blit=False, fargs=(h, p))

if (save > 0):
    anim.save ('wave.mp4', fps=30)

plt.show ()