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
import importlib

import siddons as sd
import sirt
import vir
import pyrm_funcs as pf
import visualization_tools as vt

 



#Sheep Lgan Cicular
nPix = 64
nPixels = (nPix,nPix,1)
dPix = 1.0
nDets = nPix
dDet = 1.0
nTheta = nPix*2


det_lets = 1
src_lets = 1
d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)
Thetas = np.linspace(0,np.pi,nTheta*src_lets)
Thetas = np.reshape(Thetas,(nTheta,src_lets))

srcs, trgs = sd.circular_geom_st(d.Det2_lets, Thetas, 128)
a = sd.siddons2(srcs,trgs,nPixels, dPix)
inter_pix1,inter_len1 = sd.average_inter(a)
inter_pix2 = sd.unravel_list(inter_pix1, inter_len1, nPixels)

Thetas = np.linspace(0,np.pi,nTheta)
image = shepp_logan_phantom()
image = resize(image,(nPix,nPix))

sino = sirt.forward_proj2(image, inter_pix2, inter_len2)

images2 = sirt.SIRT_py2d(nPixels[0:2], sino, inter_pix2, inter_len2, iters=10, gamma=1.0)
plt.imshow(images2[:,:,-1])


images3 = sirt.SIRT_py2d(nPixels, sino, inter_pix2, inter_len2, iters=10, gamma=1.0)
plt.imshow(images3[:,:,0,-1])















#Sheep Lgan Perp
g = pf.Geom2D()
d = pf.Detector(nPixels,dPix,dBin=1.0,nBinsY=nPix,nBinsZ=1)

srcX = nPix * dPix
DetX = -27* (nPix*dPix)/(250./.96)
DetY = d.UnqDetYs
srcY = np.add.outer(np.tan(g.Phi)*(srcX - DetX),DetY)
srcZ = 0.0
DetZ = 0.0

trg = np.stack([np.repeat(srcX,512), np.squeeze(srcY), np.repeat(srcZ,512)]).T
src = np.array([DetX,DetY[0],DetZ])
a = sd.siddons2(src,trg,nPixels, dPix)
inter_pix,inter_len = sd.siddons_list2arrays(a)




srcX = nPix * dPix
DetX = -27* (nPix*dPix)/(250./.96) - (nPix*dPix)/2.0
DetY = d.DetYs
srcY = np.add.outer(np.tan(g.Phi)*(srcX - DetX),DetY)
srcZ = 0.0
DetZ = 0.0

trg1 = np.stack([np.full(srcY.shape,srcX),srcY,np.full(srcY.shape,srcZ)]).T
src1 = np.array([np.full(srcY.shape,DetX),np.tile(DetY,[512,1]),np.full(srcY.shape,DetZ)]).T

trg2 = np.copy(trg[:,:,[1,0,2]])
src2 = np.copy(src[:,:,[1,0,2]])

trg = np.vstack([trg1,trg2])
src = np.vstack([src1,src2])


a = sd.siddons2(src,trg,nPixels, dPix)
inter_pix,inter_len = sd.siddons_list2arrays(a)

W_sino, W_img =  sirt.SIRT_weights(nPixels, inter_pix, inter_len)



image = shepp_logan_phantom()
image = resize(image,(nPix,nPix))

sino = sirt.forward_proj2(image, inter_pix, inter_len)

image = sirt.back_proj2(sino, inter_pix, inter_len,nPixels)

wow = sirt.SIRT_py2d(nPixels[0:2], sino, inter_pix, inter_len, iters=10, gamma=1.0)



W_sino, W_img = sirt.SIRT_weights(nPixels, inter_pix, inter_len)

iters = 10
gamma = 100
image_iters = np.zeros(nPixels + (iters,))
image_iters[:,:,0] = np.zeros(nPixels)

i=0
sino_est = sirt.forward_proj2(image_iters[:,:,i-1], inter_pix, inter_len)
image_update = sirt.back_proj2((sino - sino_est)/W_sino, inter_pix, inter_len, nPixels)/W_img
image_iters[:,:,i] = image_iters[:,:,i-1] + gamma*image_update





