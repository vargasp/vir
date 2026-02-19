# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:50:21 2026

@author: varga
"""




import matplotlib.pyplot as plt
import vir.sys_mat.rd as rd
import numpy as np



#Image params - Pixels
nx, ny, nz = 8, 8, 8
d_pix = 1.0

#Fan Beam Geometry - Parallel
DSO = nx/2
DSD = nx


#Sino 32 
nu, nv = 8,8
du, dv = 1., 1

img3d = np.zeros([nx,ny,nz])
img3d[15:17,15:17,15:17] = 1.0



img3d = np.zeros([nx,ny,nz])
img3d[3:5,3:5,3:5] = 1.0
sino5 = np.zeros([4,8,8,8,8], np.float32)
test1 = rd.aw_p_square(img3d, sino5,DSO, DSD,
                                 du, dv,
                                 d_pix)


img3d = np.zeros([nx,ny,nz])
img3d[3:5,3:5,3:5] = 1.0
sino5 = np.zeros([4,8,8,8,8], np.float32)
test2 = rd.aw_p_square2(img3d, sino5,DSO, DSD,
                                 du, dv,
                                 d_pix)




img3d = np.zeros([nx,ny,nz])
img3d[3:5,3:5,3:5] = 1.0
sino5 = np.zeros([4,8,8,8,8], np.float32)
test3 = rd.aw_p_square3(img3d, sino5,DSO, DSD,
                                 du, dv,
                                 d_pix)



img3d = np.zeros([nx,ny,nz])
img3d[3:5,3:5,3:5] = 1.0
sino5 = np.zeros([4,8,8,8,8], np.float32)
test4 = rd.aw_p_square4(img3d, sino5,DSO, DSD,
                                 du, dv,
                                 d_pix)




img3d = np.zeros([nx,ny,nz])
img3d[3:5,3:5,3:5] = 1.0
sino5 = np.zeros([4,8,8,8,8], np.float32)
test5 = rd.aw_p_square5(img3d, sino5,DSO, DSD,
                                 du, dv,
                                 d_pix)


