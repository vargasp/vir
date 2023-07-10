#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 21:41:39 2021

@author: vargasp
"""

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

import vir.siddon as sd
import vir.projection as proj
import vir
import vt

def sirt(sino, sdlist, W_sino, W_img, nPixels, gamma=1, iters=10):
    
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
    
    
    return image_iters


def square_geom_st(BinsX, distY, Angs):
    nBins = BinsX.size
    nAngs = Angs.size
    
    srcs = np.zeros([nBins,nAngs,4,3])
    trgs = np.zeros([nBins,nAngs,4,3])

    Bins = (nBins,nAngs)
    Bins4 = (nBins,nAngs,4)
    
    BinsY = distY
    
    
    srcs[...,0,0] = np.repeat(BinsX,nAngs).reshape(Bins)
    srcs[...,2,0] = np.repeat(BinsX,nAngs).reshape(Bins)
    srcs[...,0,1] = np.broadcast_to(-BinsY,Bins)
    srcs[...,2,1] = np.broadcast_to(BinsY,Bins)
    
    srcs[...,1,0] = np.broadcast_to(-BinsY,Bins)
    srcs[...,3,0] = np.broadcast_to(BinsY,Bins)
    srcs[...,1,1] = np.repeat(BinsX,nAngs).reshape(Bins)
    srcs[...,3,1] = np.repeat(BinsX,nAngs).reshape(Bins)

    srcs[...,2] = np.broadcast_to(0.0,Bins4)

    trgs[...,0,0] = srcs[:,:,0,0] + np.cos(Angs)*distY*5
    trgs[...,2,0] = srcs[:,:,2,0] + np.cos(Angs)*distY*5
    trgs[...,0,1] = srcs[:,:,0,1] + np.sin(Angs)*distY*5
    trgs[...,2,1] = srcs[:,:,2,1] - np.sin(Angs)*distY*5

    trgs[...,1,0] = srcs[:,:,1,0] + np.sin(Angs)*distY*5
    trgs[...,3,0] = srcs[:,:,3,0] - np.sin(Angs)*distY*5
    trgs[...,1,1] = srcs[:,:,1,1] + np.cos(Angs)*distY*5
    trgs[...,3,1] = srcs[:,:,3,1] + np.cos(Angs)*distY*5

    trgs[...,2] = srcs[...,2]

    return srcs, trgs


#Sheep Lgan Cicular
nPix = 360
nPixels = (nPix,nPix,1)
dPix = 1.1
nDets = nPix
dDet = 1.1
nTheta = nDets
det_lets = 1
src_lets = 1

gamma = 1

Dets = vir.censpace(nDets,c=0,d=dDet)
DetsDist = nPix*dPix/2.0

dTheta = np.pi/nTheta
Thetas = vir.censpace(nTheta,d=dTheta,c=np.pi/2.)

srcs, trgs = square_geom_st(Dets, DetsDist, Thetas)


sdlist = sd.siddons(srcs,trgs,nPixels, dPix).squeeze()


image = shepp_logan_phantom()
image = resize(image,(nPix,nPix))
image = image[:,:,np.newaxis]

sino  = proj.sd_f_proj(image, sdlist)
 
 
W_sino =  proj.sd_f_proj(np.ones(nPixels), sdlist)
W_img = proj.sd_b_proj(np.ones(sino.shape), sdlist, nPixels)

iters = 10
rec = sirt(sino, sdlist, W_sino, W_img, nPixels, gamma,iters=iters)


vt.CreateImage(rec[:,:,0,iters-1])











