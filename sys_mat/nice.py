# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 17:26:41 2026

@author: varga
"""


import matplotlib.pyplot as plt
import numpy as np
import vir.sys_mat.dd as dd
import time

#Half image grid + distacance to front of detector + midpoint of detecot
#(120m  + 21.38m + .01m)
DSO = 141.381
DSD = 261.381
du, dv = 2.61381, 2.61381





psino = np.load("C:\\Users\\varga\\Desktop\\sino_spheres_phantom_inter.npy")
img3d = np.load("C:\\Users\\varga\\Desktop\\phantomSpheres250.npy")

nx, ny, nz = img3d.shape
nu, nv, ns_p, ns_z, nsides = psino.shape
d_pix = .96
ds_p, ds_z = .48, .48

su, sv = 0.0, 0.0

start = time.time()

dsino1 = dd.dd_p_square(img3d,nu,nv,ns_p,ns_z,DSO,DSD,
                       du=du,dv=dv,dsrc_p=ds_p,dsrc_z=ds_z,
                       su=su,sv=sv,d_pix=1.0)

end = time.time()
print(end - start)