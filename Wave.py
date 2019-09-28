import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# 1D wave equation d2h/dt2 = c1 * d2h/dx2
n = 200
timesteps = 2000

dt = 1
dx = 1
c1 = 0.5

save = 1

uplim = 10
lowlim = -10

t1 = 1
t2 = 1

# special constant c = c1 * dt^2 / dx^2
c = c1 * dt**2 / dx**2

h = np.zeros ((timesteps, n))

# Initial conditions
width = 15
offset = 0
r = np.linspace (0, np.pi, width)
h[1,offset:offset+width] = np.sin (r)


# for t in range (2, timesteps):
#     # At the left and right boundaries
#     #h[t,0] = c * (h[t-1,0] + h[t-1,2] - 2*h[t-1,1]) \
#     #            + 2*h[t-1,0] - h[t-2,0] 

#     #h[t,n-1] = c * (h[t-1,n-3] + h[t-1,n-1] - 2*h[t-1,n-2]) \
#     #                + 2*h[t-1,n-1] - h[t-2,n-1] 

#     for i in range (1,n-1):
#         # In the middle
#         h[t,i] = c * (h[t-1,i-1] + h[t-1,i+1] - 2*h[t-1,i]) \
#                     + 2*h[t-1,i] - h[t-2,i] 



fig = plt.figure (figsize=(10,8))
ax = plt.axes (xlim = (0, n), ylim = (lowlim, uplim))
x = range (0, n, dx)
p, = ax.plot (x, h[0,:])

#title = ax.text(0.5, 0.97, "Current time: 0", bbox={'facecolor':'w', 'alpha':0.5, 'pad':2}, transform=ax.transAxes, ha="center")

def animate (t):
    # Sine left bound
    #h[t,0] = np.sin (t/10)

    if (t > 1):
        #At the left and right boundaries
        #h[t,0] = h[t-1,1]
        #h[t,n-1] = h[t-1,n-2]

        # h[t,0] = c * (h[t-1,0] + h[t-1,2] - 2*h[t-1,1]) \
        #            + 2*h[t-1,0] - h[t-2,0] 

        # h[t,n-1] = c * (h[t-1,n-3] + h[t-1,n-1] - 2*h[t-1,n-2]) \
        #                + 2*h[t-1,n-1] - h[t-2,n-1] 

        # if (t < 10):
        #     h[t,0] = 1

        for i in range (1,n-1):
            # In the middle
            h[t,i] = c * (h[t-1,i-1] + h[t-1,i+1] - 2*h[t-1,i]) \
                        + 2*t1*h[t-1,i] - t2*h[t-2,i] 

    #title.set_text("Current time: %.2f" % (t*dt))
    p.set_data (x, h[t,:])
    return p,


anim = animation.FuncAnimation (fig, animate, frames=timesteps, interval=1, blit=True, repeat=False)

if (save > 0):
    anim.save ('simple_wave.mp4', fps=60)

plt.show ()