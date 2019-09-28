import numpy as np
import matplotlib.pyplot as plt

perlresX = 500
perlresY = 500

gradresX = 10
gradresY = 10

dimensions = 2

# Random gradient field in 2D
# Add one to the dimensions, because we want to interpolate with the boundaries too
gradField = 2 * np.random.rand (gradresX+1,gradresY+1, dimensions) - 1


def fade (t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def Perlin2D (x, y):
    ### Generate the four corners of the 2D square: 
    # x0, y1    x1, y1
    # x0, y0    x1, y0

    x0 = int (x)
    x1 = x0 + 1
    y0 = int (y)
    y1 = y0 + 1

    ### Distance vectors dot product with gradient at vertices
    v00 = gradField[x0,y0,0] * (x-x0) + gradField[x0,y0,1] * (y-y0)
    v01 = gradField[x0,y1,0] * (x-x0) + gradField[x0,y1,1] * (y-y1)
    v10 = gradField[x1,y0,0] * (x-x1) + gradField[x1,y0,1] * (y-y0)
    v11 = gradField[x1,y1,0] * (x-x1) + gradField[x1,y1,1] * (y-y1)

    # Perlin noise is just interpolation with the dot product that smooths things out
    # v00 = gradField[x0,y0,0]
    # v01 = gradField[x0,y1,0]
    # v10 = gradField[x1,y0,0]
    # v11 = gradField[x1,y1,0]


    ### Interpolate value at x,y
    # Linear interpolation
    # xlower = v00 + (x-x0) * (v10 - v00)
    # xhigher = v01 + (x-x0) * (v11 - v01)

    # val = xlower + (y-y0) * (xhigher - xlower)


    # Ease curve 6t^5 - 15t^4 + 10t^3
    xlower = v00 + fade(x-x0) * (v10 - v00)
    xhigher = v01 + fade(x-x0) * (v11 - v01)

    val = xlower + fade(y-y0) * (xhigher - xlower)


    return val


perlinField = np.zeros ((perlresX, perlresY))

for i in range(perlresY):
    for j in range(perlresX):
        x = j * gradresX / perlresX
        y = i * gradresY / perlresY

        perlinField[i,j] = Perlin2D (x,y)


plt.figure (figsize=(8,8))
plt.imshow (perlinField)
plt.show ()