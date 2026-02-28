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
no, nP, nz = 5, 5, 5
d_pix = 1.0

#Fan Beam Geometry - Parallel
DSO = no/2
DSD = no


#Sino 32 
nu, nv = 5, 5
nsrc_p, nsrc_z = 5,5
du, dv = 1, 1
su, sv = 0., 0
dsrc_p, dsrc_z = 1., 1
ssrc_p, ssrc_z = 0, -2

o_bnd_arr = pf.boundspace(no,d_pix)  # vertical
p_bnd_arr = pf.boundspace(nP,d_pix)  # vertical
z_bnd_arr = pf.boundspace(nz,d_pix)  # vertical

src_z_arr = pf.censpace(nsrc_z,dsrc_z,ssrc_z)
v_bnd_arr = pf.boundspace(nv,dv,sv) + ssrc_z

M_arr = DSD /(DSO - o_bnd_arr)

# PARALLEL OVER ORTHOGONAL SLICES
inv_dv = np.float32(1.0) / dv

v0 = v_bnd_arr[0]


print(v_bnd_arr)
for iz in range(nz):
    for io in range(no):

        M = M_arr[io]

    
        proj_z_bnd_arr = M * z_bnd_arr
        proj_src_z_arr = M * src_z_arr
    
        vz_l_arr = proj_z_bnd_arr[iz]     - proj_src_z_arr + ssrc_z
        vz_r_arr = proj_z_bnd_arr[iz + 1] - proj_src_z_arr + ssrc_z
        vz_arr = (vz_r_arr+vz_l_arr)/2.
        iv_min_arr = np.clip(((vz_l_arr -v0) * inv_dv).astype(np.int32), 0, nv)



        print("io:",io,"iz:",iz,"; (",end='')
        for vz in vz_arr:
            print(f"{vz:2.2f}, ", end="")
        print("), (",end="")
        for iv_min in iv_min_arr:
            print(f"{iv_min:2d}, ", end="")
        print(")")
        
    print()

        