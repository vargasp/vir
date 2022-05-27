from siddon import *

# Assemble arguments
x1, y1, z1 = 0, 0, 0
x2, y2, z2 = 1, 1, 1
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 10, 10, 10
args = (x1, y1, z1, x2, y2, z2, dx, dy, dz, nx, ny, nz)

# Test Siddon
midpoints = midpoints(*args)
lengths = intersect_length(*args)
print('Midpoints: ' + str([(round(x[0], 3), round(x[1], 3), round(x[2], 3)) for x in midpoints]))
print('Lengths: ' + str([round(x, 3) for x in lengths]))

# Check lengths
from math import sqrt
print('Input length: ' + str(sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)))
print('Output length: ' + str(sum(lengths)))
