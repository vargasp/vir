#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 21:41:39 2021

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize
from skimage.restoration import denoise_tv_chambolle

import vir.siddon as sd
import vir.projection as proj
import vir
import vt

def sirt(sino, sdlist, W_sino, W_img, nPixels, gamma=1, beta=0, iters=10):
    
    idx1 = np.nonzero(W_sino)
    idx2 = np.nonzero(W_img)
    image_iters = np.zeros(nPixels + (iters,), dtype=np.float32)
    
    for i in range(iters):
        print("SIRT Iteration:",i)
        sino_est = proj.sd_f_proj(image_iters[...,i-1],sdlist)
        sino_est[idx1] = (sino[idx1] - sino_est[idx1])/W_sino[idx1]
        
        image_update = proj.sd_b_proj(sino_est,sdlist,nPixels)

        image_update[idx2] /= W_img[idx2]
        image_iters[...,i] = image_iters[...,i-1] + gamma*image_update
    
        if beta >0:
            image_iters[...,i] = denoise_tv_chambolle(image_iters[...,i],weight=beta)
    
    return image_iters



#Sheep Lgan Cicular
nPix = 64
nPixels = (nPix,nPix,1)
dPix = 1.
nDets = nPix
dDet = 1.
nTheta = nPix*2

gamma = 1


det_lets = 1
src_lets = 1
d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)
Thetas = np.linspace(0,np.pi,nTheta*src_lets)
Thetas = np.reshape(Thetas,(nTheta,src_lets))

srcs, trgs = sd.circular_geom_st(d.Det2_lets, Thetas)
sdlist = sd.siddons(srcs,trgs,nPixels, dPix).squeeze()

Thetas = np.linspace(0,np.pi,nTheta)
image = shepp_logan_phantom()
image = resize(image,(nPix,nPix))
image = image[:,:,np.newaxis]

sino  = proj.sd_f_proj(image, sdlist)

I0 = 1e4
I = I0*np.exp(-1* 0.167 * sino)
I =  np.random.poisson(I)
sino_n = np.log(I/I0)/ -0.167 
 
 
 
W_sino =  proj.sd_f_proj(np.ones(nPixels), sdlist)
W_img = proj.sd_b_proj(np.ones(sino.shape), sdlist, nPixels)

iters = 500
rec0 = sirt(sino_n, sdlist, W_sino, W_img, nPixels, gamma, beta=0,iters=iters)
recT = sirt(sino_n, sdlist, W_sino, W_img, nPixels, gamma, beta=1e-4,iters=iters)

#vt.CreateImage(sino)
vt.CreateImage(recT[:,:,0,iters-1])


pT = image[:,32,0]
pR = rec0[:,32,0,iters-1]
pV = recT[:,32,0,iters-1]
Ps = np.vstack([pT,pR,pV]).T
vt.CreatePlot(Ps)


