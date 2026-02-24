# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:50:21 2026

@author: varga
"""

import matplotlib.pyplot as plt
import numpy as np
import vir.sys_mat.rd as rd
import vir.sys_mat.dd as dd
import vir.sys_mat.pd as pd
import vir.sys_mat.analytic_sino as asino
from vir.sys_mat.time_testing import benchmark


#Image params - Pixels
nx, ny, nz = 32, 32, 32
d_pix = 1.0

#Fan Beam Geometry - Parallel
DSO = nx/2
DSD = nx


#Sino 32 
nu, nv = 32,32
ns_p, ns_z = 33,33
du, dv = 1, 1
su, sv = 0.0, 0.0
ds_p, ds_z = 1, 1.
 

#Phantom Paramters Sino
r = 4
x0 = 5
y0 = 0
z0 = 0

#Create analytic models
img3d= asino.phantom((x0,y0,z0,r),nx,ny,nz,upsample=5)



sino1c = dd.dd_fp_cone_3d(img3d,np.array([0]),nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
sino2c = rd.aw_fp_cone_3d(img3d,np.array([0]),nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
sino3c = rd.aw_fp_cone_3d(img3d,np.array([0]),nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix,joseph=True)


sino1s = dd.dd_p_square(img3d,nu,nv,ns_p,ns_z,DSO,DSD,
                       du=du,dv=dv,dsrc_p=ds_p,dsrc_z=ds_z,
                       su=su,sv=sv,d_pix=1.0)




sino2s = rd.aw_p_square(img3d,nu,nv,ns_p,ns_z,DSO,DSD,
                       du=du,dv=dv,ds_p=ds_p,ds_z=ds_z,
                       su=su,sv=sv,d_pix=1.0,joseph=False)

sino3s = sino2s


sinos = (sino1c[0,:,:],sino2c[0,:,:],sino3c[0,:,:],
         sino1s[0,ns_p//2,ns_z//2,:,:],sino2s[0,ns_p//2,ns_z//2,:,:],sino3s[0,ns_p//2,ns_z//2,:,:])

    
titles = ["DD Circular","SD Circular","JO Circular",
          "DD Square","SD Square","JO Square"]


plt.figure(figsize=(12,8))
for i, (sino,title) in enumerate(zip(sinos,titles)):
    plt.subplot(2,3,i+1)
    plt.imshow(sino.T, cmap='gray', aspect='auto', origin='lower')
    plt.title(title)
    if i % 3 ==0: 
        plt.ylabel("v dets")
    if i > 2:
        plt.xlabel("u dets")
plt.show()

titles = ["DD Square","SD Square","JO Square"]
sinos = (sino1s[0,:,ns_z//2,:,nv//2],sino2s[0,:,ns_z//2,:,nv//2],sino3s[0,:,ns_z//2,:,nv//2],
         sino1s[1,:,ns_z//2,:,nv//2],sino2s[1,:,ns_z//2,:,nv//2],sino3s[1,:,ns_z//2,:,nv//2],
         sino1s[2,:,ns_z//2,:,nv//2],sino2s[2,:,ns_z//2,:,nv//2],sino3s[2,:,ns_z//2,:,nv//2],
         sino1s[3,:,ns_z//2,:,nv//2],sino2s[3,:,ns_z//2,:,nv//2],sino3s[3,:,ns_z//2,:,nv//2])

plt.figure(figsize=(12,16))
for i, sino in enumerate(sinos):
    plt.subplot(4,3,i+1)
    plt.imshow(sino, cmap='gray', aspect='auto', origin='lower')
    if i < 3:
        plt.title(titles[i])
    if i % 3 ==0: 
        plt.ylabel("x")
    if i > 7:
        plt.xlabel("u Bin")
plt.show()

