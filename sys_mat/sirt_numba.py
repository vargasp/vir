#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 16:20:14 2026

@author: pvargas21
"""

import numpy as np
import vir.sys_mat.dd as dd
from numba import njit, prange


#Reconstructs the image using the SIRT algorithm
@njit(fastmath=True, cache=True)
def fast_divide1(sino, sino_est, W_sino):
    for i in range(sino.size):
        if W_sino.flat[i] != 0.0:
            sino_est.flat[i] = (sino.flat[i] - sino_est.flat[i]) / W_sino.flat[i]
    

@njit(fastmath=True, cache=True)
def fast_divide2(image_update, W_img):
    for i in range(image_update.size):
        if W_img.flat[i] != 0.0:
            image_update.flat[i] = image_update.flat[i] / W_img.flat[i]



def sirt(sino,W_sino,W_img,gem_prms,img_prms,src_prms,det_prms,srt_prms):
    DSO, DSD = gem_prms 
    nx, ny, nz, d_pix = img_prms 
    nsrc_p, nsrc_z, dsrc_p, dsrc_z, ssrc_p, ssrc_z = src_prms
    nu, nv, du, dv, su, sv = det_prms
    gamma, iters = srt_prms 
    
    img_shape = (nx, ny, nz)
    
    image_est = np.zeros(img_shape, dtype=np.float32)  + .00001
    
    
    
    #Reconstructs using SIRT
    for i in range(1,iters):
        print("SIRT Iteration:",i)
        
        #Forward projects the current image
        #Weighted projection difference
        sino_est = dd.dd_fp_square(image_est,nu,nv,nsrc_p,nsrc_z,DSO,DSD,
                                   du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                                   su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,
                                   d_pix=d_pix)

        fast_divide1(sino, sino_est, W_sino)

        #Weighted Backprojection
        image_update = dd.dd_bp_square(sino_est,(nx,ny,nz),DSO,DSD,
                                du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                                su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,
                                d_pix=d_pix)

        #Updates the image
        fast_divide2(image_update, W_img)

        image_est = image_est + gamma*image_update

      
    
    return image_est




#Phantom/Rec Parameters
img3d = np.load("/Users/pvargas21/Desktop/phantomSpheres250.npy").astype(np.float32)

nx, ny, nz = img3d.shape
d_pix = .96


#Geometric Parameters
DSO, DSD = 141.381, 261.381
nu, nv = 961, 121
du, dv = 2.61381, 2.61381
su, sv = 0.0,  158.13554
su, sv = 0.24,  158.13554


nsrc_p, nsrc_z, = 650, 10
dsrc_p, dsrc_z = .48, .48
ssrc_p, ssrc_z = 0.0, -74.4
nsides = 4

#SIRT Parameters
gamma = 1
iters = 5


gem_prms = (DSO, DSD)
img_prms = (nx, ny, nz, d_pix)
src_prms = (nsrc_p, nsrc_z, dsrc_p, dsrc_z, ssrc_p, ssrc_z)
det_prms = (nu, nv, du, dv, su, sv)
srt_prms = (gamma, iters)



#Creates the sinogram based on the true image and system matrix
sino = dd.dd_fp_square(img3d,nu,nv,nsrc_p,nsrc_z,DSO,DSD,
                       du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                       su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,
                       d_pix=d_pix)

#Creates the SIRT weights
W_sino = dd.dd_fp_square(np.ones((nx, ny, nz),np.float32),nu,nv,nsrc_p,nsrc_z,
                         DSO,DSD,
                         du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                         su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,
                         d_pix=d_pix)

#Creates the SIRT weights
W_img = dd.dd_bp_square(np.ones((nsrc_p,nsrc_z,nv,nu,4),np.float32),(nx,ny,nz),
                        DSO,DSD,
                        du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                        su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,
                        d_pix=d_pix)


#Reconstructs the image using the SIRT algorithm
rec2 = sirt(sino,W_sino,W_img,gem_prms,img_prms,src_prms,det_prms,srt_prms)










