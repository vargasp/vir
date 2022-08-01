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

import vir.analytic_geom as ag
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


def square_geom_st(BinsX, distY, Angs, sides=4):
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

    trgs[...,0,0] = srcs[:,:,0,0] + np.cos(Angs)*100
    trgs[...,2,0] = srcs[:,:,2,0] + np.cos(Angs)*100
    trgs[...,0,1] = srcs[:,:,0,1] + np.sin(Angs)*100
    trgs[...,2,1] = srcs[:,:,2,1] - np.sin(Angs)*100

    trgs[...,1,0] = srcs[:,:,1,0] + np.sin(Angs)*100
    trgs[...,3,0] = srcs[:,:,3,0] - np.sin(Angs)*100
    trgs[...,1,1] = srcs[:,:,1,1] + np.cos(Angs)*100
    trgs[...,3,1] = srcs[:,:,3,1] + np.cos(Angs)*100

    trgs[...,2] = srcs[...,2]

    for i in range(4 - sides):
        srcs = np.delete(srcs, -1, axis=-2)
        trgs = np.delete(trgs, -1, axis=-2)

    return srcs, trgs


def map_sino(srcs,trgs,dAng,dBin):
    p0 = np.array([0,0,0])
    L = ag.parametric_line(srcs,trgs)

    d = ag.line_pt_dist(L, p0)

    n = (trgs[:,:,:,0] - srcs[:,:,:,0])*(p0[1] - srcs[:,:,:,1]) - \
        (trgs[:,:,:,1] - srcs[:,:,:,1])*(p0[0] - srcs[:,:,:,0]) <1         

    d[n] = d[n]*-1


    detBins = vir.boundspace(srcs.shape[0],c=0.0,d=dBin)
    angBins = vir.boundspace(srcs.shape[1],c=np.pi,d=dAng)
    theta = np.arctan2((trgs[...,1] - srcs[...,1]),(trgs[...,0] - srcs[...,0]))

    theta = np.where(theta<0 , 2.0*np.pi+theta, theta)

    print(theta.min(), theta.max())
    return np.histogram2d(d.flatten(),theta.flatten(), bins = [detBins,angBins])[0]


#Sheep Lgan Cicular
nPix = 250
nPixels = (nPix,nPix,1)
dPix = 1.
nDets = 800
dDet = 1.0
nTheta = 360
det_lets = 1
src_lets = 1

gamma = 1

Dets = vir.censpace(nDets,c=0,d=dDet)
DetsDist = 125

dTheta = 2*np.pi/nTheta
Thetas = vir.censpace(nTheta,d=dTheta,c=np.pi)
srcs, trgs = square_geom_st(Dets, DetsDist, Thetas, sides=4)

s_map = map_sino(srcs, trgs, dTheta ,dDet).T
vt.imshow(s_map, vmax=15,xlim=(-125,125),ylim=(0,360))



vt.CreateImage(s_map, window = (0,5))
vt.hist_show(s_map, window = (0,5))




vt.CreateImage(s_map, window = (0,5), \
               coords = (Thetas[0],Thetas[-1],Dets[0], Dets[-1],  ), ctitle = " ")



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

iters = 1
rec0 = sirt(sino, sdlist, W_sino, W_img, nPixels, gamma, beta=0,iters=iters)
#recT = sirt(sino_n, sdlist, W_sino, W_img, nPixels, gamma, beta=1e-4,iters=iters)

#vt.CreateImage(sino)
vt.CreateImage(rec0[:,:,0,iters-1])









pT = image[:,32,0]
pR = rec0[:,32,0,iters-1]
pV = recT[:,32,0,iters-1]
Ps = np.vstack([pT,pR,pV]).T
vt.CreatePlot(Ps)



