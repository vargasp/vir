# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 06:53:37 2026

@author: varga
"""


import matplotlib.pyplot as plt
import numpy as np
import vir.sys_mat.dd as dd
import time
import vir.sys_mat.pf as pf





#Image params - Pixels
no, nP, nz = 4, 4, 4
d_pix = 1.0

#Fan Beam Geometry - Parallel
DSO = no/2
DSD = no


#Sino 32 
nu, nv = 4,4
nsrc_p, nsrc_z = 4,4
du, dv = 1, 1
su, sv = 0., 0
dsrc_p, dsrc_z = 1., 1
ssrc_p, ssrc_z = 0, -1.5


o_bnd_arr = pf.boundspace(no,d_pix)  # vertical
p_bnd_arr = pf.boundspace(nP,d_pix)  # vertical
z_bnd_arr = pf.boundspace(nz,d_pix)  # vertical

src_z_arr = pf.censpace(nsrc_z,dsrc_z,ssrc_z)
v_bnd_arr = pf.boundspace(nv,dv,sv)

M_arr = DSD /(DSO - o_bnd_arr)

# PARALLEL OVER ORTHOGONAL SLICES
for io in range(no):

    M = M_arr[io]

    
    proj_z_bnd_arr = M * z_bnd_arr
    proj_src_z_arr = M * src_z_arr
    
    for iz in range(nz):
        vz_l_arr = proj_z_bnd_arr[iz]     - proj_src_z_arr
        vz_r_arr = proj_z_bnd_arr[iz + 1] - proj_src_z_arr
        print("io:",io,"iz:",iz,np.around( (vz_r_arr+vz_l_arr)/2., decimals=2))
        
        
        