# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:50:21 2026

@author: varga
"""




import matplotlib.pyplot as plt
import vir.sys_mat.rd as rd
import vir.sys_mat.dd as dd
import numpy as np
from vir.sys_mat.time_testing import benchmark


import vir.sys_mat.analytic_sino as asino

#Image params - Pixels
nx, ny, nz = 32, 32, 32
d_pix = 1.0

#Fan Beam Geometry - Parallel
DSO = nx/2
DSD = nx


#Sino 32 
nu, nv = 32,32
ns_p, ns_z = 32,32
du, dv = 1., 1.
dsrc_p, dsrc_z = 1, 1.
 
img3d = np.zeros([nx,ny,nz])
img3d[15:17,15:17,15:17] = 1

img3dS = np.zeros([nx,ny,nz])
img3dS[15:17,12:14,9:11] = 1



#Phantom Paramters Sino
r = 2
x0 = 0
y0 = 0
z0 = 0

#Create analytic models
img3d = asino.phantom((x0,y0,z0,r),nx,ny,nz,upsample=5)





projS = rd.aw_fp_cone_3d(img3dS,np.array([0]),nu,nv,DSO,DSD,du=du,dv=dv,d_pix=d_pix)
plt.imshow(projS[0,:,:].T, cmap='gray', aspect='auto', origin='lower')
projC = rd.aw_fp_cone_3d(img3d,np.array([0]),nu,nv,DSO,DSD,du=du,dv=dv,d_pix=d_pix)
plt.imshow(projC[0,:,:].T, cmap='gray', aspect='auto', origin='lower')


sino5 = np.zeros([4,ns_p,ns_z,nu,nv], np.float32)

test1 = rd.aw_p_square(img3d, sino5,DSO, DSD,du, dv,d_pix)



sino5 = np.zeros([4,ns_p,ns_z,nu,nv], np.float32)
test2 = rd.aw_p_square2(img3d, sino5,DSO, DSD,du, dv,d_pix)



sino5 = np.zeros([4,ns_p,ns_z,nu,nv], np.float32)
test3 = rd.aw_p_square3(img3d,sino5,DSO, DSD,du,dv,d_pix)



sino5 = np.zeros([4,ns_p,ns_z,nu,nv], np.float32)
test4 = rd.aw_p_square4(img3d, sino5,DSO, DSD,du, dv,d_pix)

sino5 = np.zeros([4,ns_p,ns_z,nu,nv], np.float32)
test5 = rd.aw_p_square5(img3d, sino5,DSO, DSD,du, dv,d_pix)


sino5 = np.zeros([4,ns_p,ns_z,nu,nv], np.float32)
test6 = rd.aw_p_square6(img3d, sino5,DSO, DSD,du, dv, dsrc_p, dsrc_z, d_pix)



"""
benchmark(rd.aw_p_square6, img3d, sino5,DSO, DSD,du, dv,d_pix)
print(np.max(np.abs(test6 - test1)))
"""






img3d = np.zeros([nx,ny,nz])
img3d[15:17,15:17,15:17] = 1

sino5 = np.zeros([4,ns_p,ns_z,nu,nv], np.float32)
test6 = rd.aw_p_square6(img3d, sino5,DSO, DSD,du, dv, dsrc_p, dsrc_z, d_pix)
plt.imshow(test6[0,:,int(ns_z/2),:,int(nv/2)])


#plt.imshow(test6[0,int(ns_p/2),:,16,:])

x_bnd = d_pix*(np.arange(nx + 1) - nx/2).astype(np.float32)
y_bnd = d_pix*(np.arange(ny + 1) - ny/2).astype(np.float32)
z_bnd = d_pix*(np.arange(nz + 1) - nz/2).astype(np.float32)

u_bnd = du*(np.arange(nu + 1) - nu/2).astype(np.float32)
v_bnd = dv*(np.arange(nv + 1) - nv/2).astype(np.float32)

source_x = nx/2
source_y_arr = (y_bnd[:-1] + y_bnd[1:]) / 2
source_z_arr = (z_bnd[:-1] + z_bnd[1:]) / 2

sinoD= np.zeros([ns_p,ns_z,nu,nv], np.float32)
testD = dd.dd_fp_translational_0deg(
    sinoD,                # [ty, tz, u, v]
    img3d,                 # [nx, ny, nz]
    x_bnd, y_bnd, z_bnd, # voxel boundaries
    u_bnd, v_bnd,        # detector boundaries
    source_x,
    source_y_arr,        # translation in y
    source_z_arr,        # translation in z
    DSD                  # source-detector distance
)
plt.imshow(sinoD[:,int(ns_z/2),:,int(nv/2)])



"""
#Image params - Pixels
nx, ny, nz = 64, 64, 64
d_pix = 1.0

#Fan Beam Geometry - Parallel
DSO = nx/2
DSD = nx


#Sino 64 
nu, nv = 64,64
ns_p, ns_z = 64,64
du, dv = 1., 1.
dsrc_p, dsrc_z = 1., 1.
 
img3d = np.zeros([nx,ny,nz])
img3d[31:33,31:33,31:33] = 1.0


#Image params - Pixels
nx, ny, nz = 16, 16, 16
d_pix = 1.0

#Fan Beam Geometry - Parallel
DSO = nx/2
DSD = nx


#Sino 16 
nu, nv = 16,16
ns_p, ns_z = 16,16
du, dv = 1., 1

img3d = np.zeros([nx,ny,nz])
img3d[7:9,7:9,7:9] = 1.0


#Image params - Pixels
nx, ny, nz = 4, 4, 4
d_pix = 1.0

#Fan Beam Geometry - Parallel
DSO = nx/2
DSD = nx


#Sino 4 
nu, nv = 4,4
ns_p, ns_z = 4,4
du, dv = 1., 1

img3d = np.zeros([nx,ny,nz])
img3d[1:3,1:3,1:3] = 1.0
"""


#Image params - Pixels
nx, ny, nz = 5, 5, 5
d_pix = 1.0

#Fan Beam Geometry - Parallel
DSO = nx/2
DSD = nx


#Sino 4 
nu, nv = 5,5
ns_p, ns_z = 5,5
du, dv = 1., 1

img3d = np.ones([nx,ny,nz])
img3d[1:4,1:4,1:4] = 1.0