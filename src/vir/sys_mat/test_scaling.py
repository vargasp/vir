
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 14:02:51 2026

@author: pvargas21
"""


import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600
import numpy as np

import vir.sys_mat.dd as dd
import vir.sys_mat.rd as rd
import vir.sys_mat.pd as pd


#Image params - Pixels
nx = 128
ny = 128
nz = 128
d_pix = 1


#Fan Beam Geometry - Parallel
DSO = 1e8
DSD = 1e8 + max(nx,ny)/2

#Fan Beam Geometry - Parallel
DSO = max(nx,ny)*np.sqrt(2)/2 *d_pix
DSD = DSO*2



#Sino params 
na = 2
nu = 128
nv = 128
du = 1
dv = 1
su = 0.0
sv = 0.0

ang_arr = np.array([0,np.pi/4])
u_arr = du*(np.arange(nu) - nu/2.0 + 0.5 + su)




#Test image
img3d = np.zeros((nx, ny, nz), dtype=np.float32)
#img3d[62:66, 40:88, 40:88] = 1.0  # center impulse
img3d[59:69, 39:89, 39:89] = 1.0  # center impulse
#img3d[10:22, 10:22, 10:22] = 1.0  # center impulse


img2d = img3d[:, :, int(nz/2)] 

#sino1p = dd.dd_fp_par_2d(img2d, ang_arr, nu, du=du, su=su, d_pix=d_pix)
#sino2p = rd.aw_fp_par_2d(img2d, ang_arr, nu, du=du, su=su, d_pix=d_pix)
#sino3p = rd.aw_fp_par_2d(img2d, ang_arr, nu, du=du, su=su, d_pix=d_pix,joseph=True)
#sino4p = pd.pd_fp_par_2d(img2d, ang_arr, nu, du=du, su=su, d_pix=d_pix)

sino1f = dd.dd_fp_fan_2d(img2d, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix)
sino2f = rd.aw_fp_fan_2d(img2d, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix)
sino3f = rd.aw_fp_fan_2d(img2d, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix,joseph=True)
sino4f = pd.pd_fp_fan_2d(img2d, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix)

#sino1c = dd.dd_fp_cone_3d(img3d, ang_arr, nu, nv, DSO, DSD, du=du, dv=dv,su=su, sv=sv, d_pix=d_pix)
#sino2c = rd.aw_fp_cone_3d(img3d, ang_arr, nu, nv, DSO, DSD, du=du, d_pix=d_pix)
#sino3c = rd.aw_fp_cone_3d(img3d, ang_arr, nu, nv, DSO, DSD, du=du, d_pix=d_pix,joseph=True)
#sino4c = pd.pd_fp_cone_3d(img3d, ang_arr, nu, nv, DSO, DSD, du=du, dv=dv,su=su, sv=sv,d_pix=d_pix)



"""
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(sino1p[0,:], label='DD')
plt.plot(sino2p[0,:], label='AW')
plt.plot(sino3p[0,:], label='JO')
plt.plot(sino4p[0,:], label='PD')
plt.legend()
plt.title("u det: Angle 0 profile")

plt.subplot(1,2,2)
plt.plot(sino1p[1,:], label='DD')
plt.plot(sino2p[1,:], label='AW')
plt.plot(sino3p[1,:], label='JO')
plt.plot(sino4p[1,:], label='PD')
plt.legend()
plt.title("udet: Angle 45 profile")
plt.show()
"""

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(sino1f[0,:], label='DD')
plt.plot(sino2f[0,:], label='AW')
plt.plot(sino3f[0,:], label='JO')
plt.plot(sino4f[0,:], label='PD')
plt.legend()
plt.title("u det: Angle 0 profile")

plt.subplot(1,2,2)
plt.plot(sino1f[1,:], label='DD')
plt.plot(sino2f[1,:], label='AW')
plt.plot(sino3f[1,:], label='JO')
plt.plot(sino4f[1,:], label='PD')
plt.legend()
plt.title("udet: Angle 45 profile")
plt.show()



"""
plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.plot(sino1c[0,:,int(nz/2)], label='DD')
plt.plot(sino2c[0,:,int(nz/2)], label='AW')
plt.plot(sino3c[0,:,int(nz/2)], label='JO')
plt.plot(sino4c[0,:,int(nz/2)], label='PD')
plt.legend()
plt.title("u det: Angle 0 profile")

plt.subplot(2,2,2)
plt.plot(sino1c[0,int(ny/2),:], label='DD')
plt.plot(sino2c[0,int(ny/2),:], label='AW')
plt.plot(sino3c[0,int(ny/2),:], label='JO')
plt.plot(sino4c[0,int(ny/2),:], label='PD')
plt.legend()
plt.title("v det: Angle 0 profile")

plt.subplot(2,2,3)
plt.plot(sino1c[1,:,int(nz/2)], label='DD')
plt.plot(sino2c[1,:,int(nz/2)], label='AW')
plt.plot(sino3c[1,:,int(nz/2)], label='JO')
plt.plot(sino4c[1,:,int(nz/2)], label='PD')
plt.legend()
plt.title("udet: Angle 45 profile")


plt.subplot(2,2,4)
plt.plot(sino1c[1,int(ny/2),:], label='DD')
plt.plot(sino2c[1,int(ny/2),:], label='AW')
plt.plot(sino3c[1,int(ny/2),:], label='JO')
plt.plot(sino4c[1,int(ny/2),:], label='PD')
plt.legend()
plt.title("v det: Angle 45 profile")
plt.show()
"""
