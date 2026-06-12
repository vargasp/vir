import numpy as np
import matplotlib.pyplot as plt

import vir
import vir.sys_mat.analytic_sino as asino

na = 512 
nu = 256
nv = 256
angles = np.linspace(0,2*np.pi,na,endpoint=False)
u = vir.censpace(nu)
v = vir.censpace(nv)

DSO = 1000.0
DSD = 1500.0

s = (14,14,10,1.0)

test2par = asino.analytic_circle_sino_par_2d(s,angles,u)
test2fan = asino.analytic_circle_sino_fan_2d(s, angles, u, DSO, DSD)
test3a = asino.analytic_sphere_sino_cone_3d(s, angles, u, v, DSO, DSD)
test3b = asino.analytic_sphere_sino_cone_3d0(s, angles, u, v, DSO, DSD)

"""
test = asino.sphere_projection(src, det, sphere, rho=1.0):


sphere_projection_gauss(
        src_centers,      # (...,3) array of source positions with arbitrary leading dimensions
        det_centers,      # (Nu,Nv,3) detector pixel centers
        eu, ev,           # detector basis vectors (3,)
        du, dv,           # pixel size
        sphere,
        src_size=(0,0),   # focal spot size (sy, sz)
        src_nodes=2,      # Gaussian nodes for source/focal spot
        det_nodes=2,      # Gaussian nodes per detector pixel
        rho=1.0
    ):
"""

print(test3a.shape)
plt.imshow(test3a[0,:,:],aspect='auto')
plt.show()