# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:01:03 2026

@author: varga
"""

import matplotlib.pyplot as plt
import numpy as np
import vir.sys_mat.analytic_sino as asino
import vir.sys_mat.pf as pf


import vir.sys_mat.dd as dd

#Image params - Pixels
nx, ny, nz = 17, 17, 17

d_pix = 1.0 

#Phantom Paramters Sino
r = 4
x0 = 0
y0 = 0
z0 = 0

sphere = (x0,y0,z0,r)



#Create analytic models
img = asino.sphere_phantom_exact((x0,y0,z0,r/d_pix), nx, ny, nz)

"""
plt.figure(figsize=(12,3))

#Y-Z plane
plt.subplot(1,3,1)
plt.imshow(img[int(nx/2+x0),:,:].T, cmap='gray', aspect='auto', origin='lower')
plt.title("YZ Plane")
plt.xlabel("y")
plt.ylabel("z")

#X-Z plane
plt.subplot(1,3,2)
plt.imshow(img[:,int(ny/2+y0),:].T, cmap='gray', aspect='auto', origin='lower')
plt.title("XZ Plane")
plt.xlabel("x")
plt.ylabel("z")

#X-Y plane
plt.subplot(1,3,3)
plt.imshow(img[:,:,int(nz/2+z0)].T, cmap='gray', aspect='auto', origin='lower')
plt.title("XY Plane")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
"""



#Fan Beam Geometry - Parallel
DSO = d_pix*nx/2
DSD = DSO + d_pix*nx/2

#Sino 32 
nu, nv = 33,33
nsrc_p, nsrc_z = 33,33
du, dv = 1., 1.
su, sv = 0., 0.
dsrc_p, dsrc_z = 1, 1.
ssrc_p, ssrc_z = 5., 0.

src_p_arr = pf.censpace(nsrc_p,dsrc_p,ssrc_p)
src_z_arr = pf.censpace(nsrc_z,dsrc_z,ssrc_z)
u_arr = pf.censpace(nu,du,su)
v_arr = pf.censpace(nv,dv,sv)

U, V = np.meshgrid(u_arr, v_arr, indexing="ij")
dets = np.zeros([nu,nv,3])
dets[:,:,0] = DSO
dets[:,:,1] = U
dets[:,:,2] = V


Src_p, Src_z = np.meshgrid(src_p_arr, src_z_arr, indexing='ij')
srcs = np.zeros((nsrc_p,nsrc_z,3))
srcs[...,0] = -DSO
srcs[...,1] = Src_p
srcs[...,2] = Src_z






# Compute sinogram
sino = np.empty([nsrc_p,nsrc_z,nu,nv,4])


#All
srcs[...,2] = Src_z
ev = np.array((0,0,1))

#positive X (from -y to +y)
eu = np.array((1,0,0))
srcs[...,0] = DSO
srcs[...,1] = Src_p
dets[...,0] = DSO-DSD
for i, src_p in enumerate(src_p_arr):
    dets[:,:,1] = U + src_p
    for j, src_z in enumerate(src_z_arr):
        dets[:,:,2] = V + src_z
        
        sino[i,j,:,:,0] = asino.sphere_projection_gauss(srcs[i,j,:],dets,eu, ev,du, dv,sphere,
                                      src_size=(dsrc_p,dsrc_z), 
                                      src_nodes=2,det_nodes=2,rho=1.0)


#positive Y (from -x to +x)
eu = np.array((0,1,0))
srcs[...,1] = DSO
srcs[...,0] = Src_p
dets[...,1] = DSO-DSD
for i, src_p in enumerate(src_p_arr):
    dets[:,:,0] = -U + src_p
    for j, src_z in enumerate(src_z_arr):
        dets[:,:,2] = V + src_z
        
        sino[i,j,:,:,1] = asino.sphere_projection_gauss(srcs[i,j,:],dets,eu, ev,du, dv,sphere,
                                      src_size=(dsrc_p,dsrc_z), 
                                      src_nodes=2,det_nodes=2,rho=1.0)


#negative X (from -y to +y)
eu = np.array((1,0,0))
srcs[...,0] = -DSO
srcs[...,1] = Src_p
dets[...,0] = DSD-DSO
for i, src_p in enumerate(src_p_arr):
    dets[:,:,1]= -U + src_p
    for j, src_z in enumerate(src_z_arr):
        dets[:,:,2] = V + src_z
        
        sino[i,j,:,:,2] = asino.sphere_projection_gauss(srcs[i,j,:],dets,eu, ev,du, dv,sphere,
                                      src_size=(dsrc_p,dsrc_z), 
                                      src_nodes=2,det_nodes=2,rho=1.0)


#positive Y (from -x to +x)
eu = np.array((0,1,0))
srcs[...,0] = Src_p
srcs[...,1] = -DSO
dets[...,1] = DSD-DSO
for i, src_p in enumerate(src_p_arr):
    dets[:,:,0] = U + src_p
    for j, src_z in enumerate(src_z_arr):
        dets[:,:,2] = V + src_z
        
        sino[i,j,:,:,3] = asino.sphere_projection_gauss(srcs[i,j,:],dets,eu, ev,du, dv,sphere,
                                      src_size=(dsrc_p,dsrc_z), 
                                      src_nodes=2,det_nodes=2,rho=1.0)



#plt.plot(sino[nsrc_p//2,nsrc_z//2,:,nv//2,0],label = "Analtic 0 ")
#plt.plot(sino[nsrc_p//2,nsrc_z//2,:,nv//2,1],label = "Analtic 1")
#plt.plot(sino[nsrc_p//2,nsrc_z//2,:,nv//2,0],label = "Analtic 2")
#plt.plot(sino[nsrc_p//2,nsrc_z//2,:,nv//2,1],label = "Analtic 3")
#plt.title("Intersection Length")
#plt.ylabel("Intersection Length")
#plt.xlabel("u dets")
#plt.legend()
#plt.show()




sino1s = dd.dd_fp_square(img,nu,nv,nsrc_p,nsrc_z,DSO,DSD,
                       du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                       su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,d_pix=d_pix).transpose([0,1,3,2,4])/d_pix



plt.figure(figsize=(16,8))
plt.subplot(2,4,1)
plt.imshow(sino[:,:,nu//2,nv//2,0].T, cmap='gray', aspect='auto', origin='lower',
           extent = (src_p_arr[0],src_p_arr[-1],src_z_arr[0],src_z_arr[-1]))
plt.title("Side 0")
plt.ylabel("z srcs")
plt.xlabel("y srcs")

plt.subplot(2,4,2)
plt.imshow(sino[:,:,nu//2,nv//2,1].T, cmap='gray', aspect='auto', origin='lower',
           extent = (src_p_arr[0],src_p_arr[-1],src_z_arr[0],src_z_arr[-1]))
plt.title("Side 1")
plt.ylabel("z srcs")
plt.xlabel("x srcs")

plt.subplot(2,4,3)
plt.imshow(sino[:,:,nu//2,nv//2,2].T, cmap='gray', aspect='auto', origin='lower',
           extent = (src_p_arr[0],src_p_arr[-1],src_z_arr[0],src_z_arr[-1]))
plt.title("Side 2")
plt.ylabel("z srcs")
plt.xlabel("y srcs")

plt.subplot(2,4,4)
plt.imshow(sino[:,:,nu//2,nv//2,3].T, cmap='gray', aspect='auto', origin='lower',
           extent = (src_p_arr[0],src_p_arr[-1],src_z_arr[0],src_z_arr[-1]))
plt.title("Side 3")
plt.ylabel("z srcs")
plt.xlabel("x srcs")

u_ang = np.arctan((u_arr-ssrc_p)/DSD)*180/np.pi
v_ang = np.arctan((v_arr-ssrc_z)/DSD)*180/np.pi

plt.subplot(2,4,5)
plt.imshow(sino[nsrc_p//2,nsrc_z//2,:,:,0].T, cmap='gray', aspect='auto', origin='lower',
           extent = (u_ang[0],u_ang[-1],v_ang[0],v_ang[-1]))
plt.ylabel("v dets")
plt.xlabel("u dets")

plt.subplot(2,4,6)
plt.imshow(sino[nsrc_p//2,nsrc_z//2,:,:,1].T, cmap='gray', aspect='auto', origin='lower',
           extent = (u_ang[0],u_ang[-1],v_ang[0],v_ang[-1]))
plt.ylabel("v dets")
plt.xlabel("u dets")

plt.subplot(2,4,7)
plt.imshow(sino[nsrc_p//2,nsrc_z//2,:,:,2].T, cmap='gray', aspect='auto', origin='lower',
           extent = (u_ang[0],u_ang[-1],v_ang[0],v_ang[-1]))
plt.ylabel("v dets")
plt.xlabel("u dets")

plt.subplot(2,4,8)
plt.imshow(sino[nsrc_p//2,nsrc_z//2,:,:,3].T, cmap='gray', aspect='auto', origin='lower',
           extent = (u_ang[0],u_ang[-1],v_ang[0],v_ang[-1]))
plt.ylabel("v dets")
plt.xlabel("u dets")
plt.show



plt.figure(figsize=(16,8))
plt.subplot(2,4,1)
plt.imshow(sino1s[:,:,nu//2,nv//2,0].T, cmap='gray', aspect='auto', origin='lower',
           extent = (src_p_arr[0],src_p_arr[-1],src_z_arr[0],src_z_arr[-1]))
plt.title("Side 0")
plt.ylabel("z srcs")
plt.xlabel("y srcs")

plt.subplot(2,4,2)
plt.imshow(sino1s[:,:,nu//2,nv//2,1].T, cmap='gray', aspect='auto', origin='lower',
           extent = (src_p_arr[0],src_p_arr[-1],src_z_arr[0],src_z_arr[-1]))
plt.title("Side 1")
plt.ylabel("z srcs")
plt.xlabel("x srcs")

plt.subplot(2,4,3)
plt.imshow(sino1s[:,:,nu//2,nv//2,2].T, cmap='gray', aspect='auto', origin='lower',
           extent = (src_p_arr[0],src_p_arr[-1],src_z_arr[0],src_z_arr[-1]))
plt.title("Side 2")
plt.ylabel("z srcs")
plt.xlabel("y srcs")

plt.subplot(2,4,4)
plt.imshow(sino1s[:,:,nu//2,nv//2,3].T, cmap='gray', aspect='auto', origin='lower',
           extent = (src_p_arr[0],src_p_arr[-1],src_z_arr[0],src_z_arr[-1]))
plt.title("Side 3")
plt.ylabel("z srcs")
plt.xlabel("x srcs")


plt.subplot(2,4,5)
plt.imshow(sino1s[nsrc_p//2,nsrc_z//2,:,:,0].T, cmap='gray', aspect='auto', origin='lower',
           extent = (u_ang[0],u_ang[-1],v_ang[0],v_ang[-1]))
plt.ylabel("v dets")
plt.xlabel("u dets")

plt.subplot(2,4,6)
plt.imshow(sino1s[nsrc_p//2,nsrc_z//2,:,:,1].T, cmap='gray', aspect='auto', origin='lower',
           extent = (u_ang[0],u_ang[-1],v_ang[0],v_ang[-1]))
plt.ylabel("v dets")
plt.xlabel("u dets")

plt.subplot(2,4,7)
plt.imshow(sino1s[nsrc_p//2,nsrc_z//2,:,:,2].T, cmap='gray', aspect='auto', origin='lower',
           extent = (u_ang[0],u_ang[-1],v_ang[0],v_ang[-1]))
plt.ylabel("v dets")
plt.xlabel("u dets")

plt.subplot(2,4,8)
plt.imshow(sino1s[nsrc_p//2,nsrc_z//2,:,:,3].T, cmap='gray', aspect='auto', origin='lower',
           extent = (u_ang[0],u_ang[-1],v_ang[0],v_ang[-1]))
plt.ylabel("v dets")
plt.xlabel("u dets")
plt.show


"""
plt.figure(figsize=(4,4))
plt.subplot(1,1,1)
plt.plot(sino[nsrc_p//2,nsrc_z//2,:,nv//2,0],label = "Analtic 0")
plt.plot(sino1s[nsrc_p//2,nsrc_z//2,:,nv//2,0],label = "DD 0 ")
plt.plot(sino[nsrc_p//2,nsrc_z//2,:,nv//2,1],label = "Analtic 1")
plt.plot(sino1s[nsrc_p//2,nsrc_z//2,:,nv//2,1],label = "DD 1")
plt.title("Intersection Length")
plt.ylabel("Intersection Length")
plt.xlabel("u dets")
plt.legend()
plt.show()


plt.figure(figsize=(4,4))
plt.subplot(1,1,1)

plt.plot(sino[:,nsrc_z//2,nu//2,nv//2,0],label = "Analtic 0")
plt.plot(sino1s[:,nsrc_z//2,nu//2,nv//2,0],label = "DD 0 ")
plt.plot(sino[:,nsrc_z//2,nu//2,nv//2,1],label = "Analtic 1")
plt.plot(sino1s[:,nsrc_z//2,nu//2,nv//2,1],label = "DD 1")
plt.title("Intersection Length")
plt.ylabel("Intersection Length")
plt.xlabel("x dets")
plt.legend()
plt.show()

"""

