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
nu, nv = 64,64
ns_p, ns_z = 32,32
du, dv = 1, 1
su, sv = 0.0, 0.0
ds_p, ds_z = 1., 1.
 

#Phantom Paramters Sino
r = 4
x0 = 4
y0 = 2
z0 = 0

#Create analytic models
img3d= asino.phantom((x0,y0,z0,r),nx,ny,nz,upsample=5)

#Conebeam
sino1c = dd.dd_fp_cone_3d(img3d,np.array([0]),nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
sino2c = rd.aw_fp_cone_3d(img3d,np.array([0]),nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
sino3c = rd.aw_fp_cone_3d(img3d,np.array([0]),nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix,joseph=True)


#Square
sino1s = dd.dd_fp_square(img3d,nu,nv,ns_p,ns_z,DSO,DSD,
                       du=du,dv=dv,dsrc_p=ds_p,dsrc_z=ds_z,
                       su=su,sv=sv,d_pix=1.0)
sino2s = rd.aw_p_square(img3d,nu,nv,ns_p,ns_z,DSO,DSD,
                       du=du,dv=dv,ds_p=ds_p,ds_z=ds_z,
                       su=su,sv=sv,d_pix=1.0,joseph=False)


#[ns_p,np_z,nu,nv,4]
sino1s = sino1s.transpose([0,1,3,2,4,])

#[ns_p,np_z,nu,nv,4]
sino2s = sino2s.transpose([1,2,3,4,0])

sino3s = sino2s



"""
Compare forward projection conebeam and square
"""
sinos = (sino1c[0,:,:],sino2c[0,:,:],sino3c[0,:,:],
         sino1s[ns_p//2,ns_z//2,:,:,0],sino2s[ns_p//2,ns_z//2,:,:,0],sino3s[ns_p//2,ns_z//2,:,:,0])

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
sinos = (sino1s[:,ns_z//2,:,nv//2,0],sino2s[:,ns_z//2,:,nv//2,0],sino3s[:,ns_z//2,:,nv//2,0],
         sino1s[:,ns_z//2,:,nv//2,1],sino2s[:,ns_z//2,:,nv//2,1],sino3s[:,ns_z//2,:,nv//2,1],
         sino1s[:,ns_z//2,:,nv//2,2],sino2s[:,ns_z//2,:,nv//2,2],sino3s[:,ns_z//2,:,nv//2,2],
         sino1s[:,ns_z//2,:,nv//2,3],sino2s[:,ns_z//2,:,nv//2,3],sino3s[:,ns_z//2,:,nv//2,3])

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



"""
Examine bacprojection sides

"""

sino1s = np.ascontiguousarray(sino1s.transpose([0,1,3,2,4,]))

sino0 = sino1s.copy()
sino1 = sino1s.copy()
sino2 = sino1s.copy()
sino3 = sino1s.copy()
sino0[...,1:] = 0.0
sino1[...,0] = 0.0
sino1[...,2:] = 0.0
sino2[...,:2] = 0.0
sino2[...,3:] = 0.0
sino3[...,:3] = 0.0

rec0 = dd.dd_bp_square(sino0,(nx,ny,nz), DSO,DSD,
                       du=du,dv=dv,dsrc_p=ds_p,dsrc_z=ds_z,
                       su=su,sv=sv,d_pix=1.0)

rec1 = dd.dd_bp_square(sino1,(nx,ny,nz), DSO,DSD,
                       du=du,dv=dv,dsrc_p=ds_p,dsrc_z=ds_z,
                       su=su,sv=sv,d_pix=1.0)

rec2 = dd.dd_bp_square(sino2,(nx,ny,nz), DSO,DSD,
                       du=du,dv=dv,dsrc_p=ds_p,dsrc_z=ds_z,
                       su=su,sv=sv,d_pix=1.0)

rec3 = dd.dd_bp_square(sino3,(nx,ny,nz), DSO,DSD,
                       du=du,dv=dv,dsrc_p=ds_p,dsrc_z=ds_z,
                       su=su,sv=sv,d_pix=1.0)

recS = dd.dd_bp_square(sino1s,(nx,ny,nz), DSO,DSD,
                       du=du,dv=dv,dsrc_p=ds_p,dsrc_z=ds_z,
                       su=su,sv=sv,d_pix=1.0)

recs = [rec0,rec1,rec2,rec3,recS]
plt.figure(figsize=(12,20))

for i, rec in enumerate(recs):
    #Y-Z plane
    plt.subplot(5,3,(i*3)+1)
    plt.imshow(rec[int(nx/2+x0),:,:].T, cmap='gray', aspect='auto', origin='lower')
    plt.xlabel("y")
    plt.ylabel("z")

    #X-Z plane
    plt.subplot(5,3,(i*3)+2)
    plt.imshow(rec[:,int(ny/2+y0),:].T, cmap='gray', aspect='auto', origin='lower')
    plt.xlabel("x")
    plt.ylabel("z")
    
    #X-Y plane
    plt.subplot(5,3,(i*3)+3)
    plt.imshow(rec[:,:,int(nz/2+z0)].T, cmap='gray', aspect='auto', origin='lower')
    plt.xlabel("x")
    plt.ylabel("y")
plt.show()

