# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:50:21 2026

@author: varga
"""




import matplotlib.pyplot as plt
import vir.sys_mat.rd as rd
import numpy as np



img3d = np.zeros([nx,ny,nz])
img3d[15:17,15:17,15:17] = 1.0



#Image params - Pixels
nx, ny, nz = 16, 16, 16
d_pix = 1.0

#Fan Beam Geometry - Parallel
DSO = nx/2
DSD = nx


#Sino 32 
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


#Sino 32 
nu, nv = 4,4
ns_p, ns_z = 4,4
du, dv = 1., 1

img3d = np.zeros([nx,ny,nz])
img3d[1:3,1:3,1:3] = 1.0

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
test6 = rd.aw_p_square6(img3d, sino5,DSO, DSD,du, dv,d_pix)


