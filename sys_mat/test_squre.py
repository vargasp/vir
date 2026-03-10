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
import vir.sys_mat.pf as pf


#Image params - Pixels
nx, ny, nz = 33, 33, 33
d_pix = 1.0

#Fan Beam Geometry - Parallel
DSO = nx/2
DSD = nx


#Sino 32 
nu, nv = 65,65
nsrc_p, nsrc_z = 65,65
du, dv = 1, 1
su, sv = 0, 0.
dsrc_p, dsrc_z = 1., 1.
ssrc_p, ssrc_z = 0., 0.

z_bnd_arr = pf.boundspace(nz,d_pix)  # vertical
src_z_arr = pf.censpace(nsrc_z,dsrc_z,ssrc_z)
v_bnd_arr = pf.boundspace(nv,dv,sv + ssrc_z)




#Phantom Paramters Sino
r = 5
x0 = 0
y0 = 0
z0 = 0

#Create analytic models
#img3d= asino.phantom((x0,y0,z0,r),nx*3,ny*3,nz*3,upsample=5)
img3d= asino.sphere_phantom_exact((x0,y0,z0,r),nx,ny,nz)



#Conebeam
#sino1c = dd.dd_fp_cone_3d(img3d,np.array([0]),nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
#sino2c = rd.aw_fp_cone_3d(img3d,np.array([0]),nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
#sino3c = rd.aw_fp_cone_3d(img3d,np.array([0]),nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix,joseph=True)


#Square
sino1s = dd.dd_fp_square(img3d,nu,nv,nsrc_p,nsrc_z,DSO,DSD,
                       du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                       su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,d_pix=d_pix)


#sino2s = rd.aw_p_square(img3d,nu,nv,ns_p,ns_z,DSO,DSD,
#                       du=du,dv=dv,ds_p=ds_p,ds_z=ds_z,
#                       su=su,sv=sv,d_pix=1.0,joseph=False)


#[ns_p,np_z,nu,nv,4]
sino1s = sino1s.transpose([0,1,3,2,4])

#[ns_p,np_z,nu,nv,4]
#sino2s = sino2s.transpose([1,2,3,4,0])

#sino3s = sino2s




plt.figure(figsize=(16,8))
plt.subplot(2,4,1)
plt.imshow(sino1s[:,:,nu//2,nv//2,0].T, cmap='gray', aspect='auto', origin='lower')
plt.title("Side 0")
plt.ylabel("z srcs")
plt.xlabel("y srcs")

plt.subplot(2,4,2)
plt.imshow(sino1s[:,:,nu//2,nv//2,1].T, cmap='gray', aspect='auto', origin='lower')
plt.title("Side 1")
plt.ylabel("z srcs")
plt.xlabel("x srcs")

plt.subplot(2,4,3)
plt.imshow(sino1s[:,:,nu//2,nv//2,2].T, cmap='gray', aspect='auto', origin='lower')
plt.title("Side 2")
plt.ylabel("z srcs")
plt.xlabel("y srcs")

plt.subplot(2,4,4)
plt.imshow(sino1s[:,:,nu//2,nv//2,3].T, cmap='gray', aspect='auto', origin='lower')
plt.title("Side 3")
plt.ylabel("z srcs")
plt.xlabel("x srcs")


plt.subplot(2,4,5)
plt.imshow(sino1s[nsrc_p//2,nsrc_z//2,:,:,0].T, cmap='gray', aspect='auto', origin='lower')
plt.ylabel("v dets")
plt.xlabel("u dets")

plt.subplot(2,4,6)
plt.imshow(sino1s[nsrc_p//2,nsrc_z//2,:,:,1].T, cmap='gray', aspect='auto', origin='lower')
plt.ylabel("v dets")
plt.xlabel("u dets")

plt.subplot(2,4,7)
plt.imshow(sino1s[nsrc_p//2,nsrc_z//2,:,:,2].T, cmap='gray', aspect='auto', origin='lower')
plt.ylabel("v dets")
plt.xlabel("u dets")

plt.subplot(2,4,8)
plt.imshow(sino1s[nsrc_p//2,nsrc_z//2,:,:,3].T, cmap='gray', aspect='auto', origin='lower')
plt.ylabel("v dets")
plt.xlabel("u dets")
plt.show

plt.plot(sino1s[nsrc_p//2,nsrc_z//2,52:77,nv//2,:])
plt.plot(sino1s[nsrc_p//2,58:71,nu//2,nv//2,:])




#Compare forward projection square
"""
sinos = (sino1s[ns_p//2,ns_z//2,:,:,0],sino2s[ns_p//2,ns_z//2,:,:,0],sino3s[ns_p//2,ns_z//2,:,:,0])

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














#Compare forward projection conebeam and square

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


#Examine bacprojection sides


sino1s = sino1s.transpose([0,1,3,2,4])

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
                       du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                       su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,d_pix=d_pix)

rec1 = dd.dd_bp_square(sino1,(nx,ny,nz), DSO,DSD,
                       du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                       su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,d_pix=d_pix)

rec2 = dd.dd_bp_square(sino2,(nx,ny,nz), DSO,DSD,
                       du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                       su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,d_pix=d_pix)

rec3 = dd.dd_bp_square(sino3,(nx,ny,nz), DSO,DSD,
                       du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                       su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,d_pix=d_pix)

recS = dd.dd_bp_square(sino1s,(nx,ny,nz), DSO,DSD,
                       du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                       su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,d_pix=d_pix)



recs = [rec0,rec1,rec2,rec3,recS]
titles = ["Side 0","Side 1","Side 2","Side 3","All sides"]
plt.figure(figsize=(20,12))

for i, rec in enumerate(recs):       
    #Y-Z plane
    plt.subplot(3,5,i+1)
    plt.imshow(rec[int(nx/2+x0),:,:].T, cmap='gray', aspect='auto', origin='lower')
    plt.title(titles[i])
    plt.xlabel("y")
    plt.ylabel("z")

for i, rec in enumerate(recs):       
    #X-Z plane
    plt.subplot(3,5,5 + i+1)
    plt.imshow(rec[:,int(ny/2+y0),:].T, cmap='gray', aspect='auto', origin='lower')
    plt.xlabel("x")
    plt.ylabel("z")

for i, rec in enumerate(recs):       
    #X-Y plane
    plt.subplot(3,5,10+i+1)
    plt.imshow(rec[:,:,int(nz/2+z0)].T, cmap='gray', aspect='auto', origin='lower')
    plt.xlabel("x")
    plt.ylabel("y")
plt.show()




su_s = np.array([0,.25,.52,.75,1])

prof_x = np.zeros((nx, su_s.size))
prof_y = np.zeros((nx, su_s.size))
prof_z = np.zeros((nx, su_s.size))

for i, su in enumerate(su_s):
    sino1s = dd.dd_fp_square(img3d,nu,nv,nsrc_p,nsrc_z,DSO,DSD,
                           du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                           su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,d_pix=d_pix)
    
    
    recS = dd.dd_bp_square(sino1s,(nx,ny,nz), DSO,DSD,
                           du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                           su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,d_pix=d_pix)

    prof_x[:,i] = recS[:,int(ny/2+y0),int(nz/2+z0)]
    prof_y[:,i] = recS[int(nx/2+x0),:,int(nz/2+z0)]
    prof_z[:,i] = recS[int(nx/2+x0),int(ny/2+y0),:]


"""


plt.plot(prof_x[5:-5])
         


         t="x-axis")
plt.plot(rec[int(nx/2+x0),:,int(nz/2+z0)],label="y-axis")
plt.plot(rec[int(nx/2+x0),int(ny/2+y0),:],label="z-axis")
plt.legend()


"""



