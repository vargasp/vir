#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 21:41:39 2021

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

import siddons as sd
import sirt
import vir
import visualization_tools as vt


def shepp_sirt(nPix,dPix,nTheta,det_lets,src_lets,ave_axes=None):
    nPixels = (nPix,nPix,1)
    nDets = nPix*2
    dDet = dPix/2.0
  

    d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)
    Thetas = np.linspace(0,np.pi,nTheta*src_lets)
    Thetas = np.reshape(Thetas,(nTheta,src_lets))

    srcs, trgs = sd.circular_geom_st(d.Det2_lets, Thetas, 128)
    a = sd.siddons2(srcs,trgs,nPixels, dPix)
    inter_pix,inter_len = sd.average_inter(a,ave_axes=ave_axes)

    Thetas = np.linspace(0,np.pi,nTheta)

    image = shepp_logan_phantom()
    image = resize(image,(nPix,nPix))

    sino = sirt.forward_proj2(image, inter_pix, inter_len)

    images = sirt.SIRT_py2d(nPixels[0:2],sino,inter_pix,inter_len,\
                             iters=400,gamma=1.0)
    return images[:,:,-1]





nPix = 64
dPix = 1
nTheta = nPix*2
d_lets = 1
s_lets = 1


true = shepp_logan_phantom()
true = resize(true,(nPix,nPix))
plt.imshow(true)


image = shepp_sirt(nPix,dPix,nTheta,d_lets,s_lets)
plt.imshow(image)


image2 = shepp_sirt(nPix,dPix,nTheta,d_lets,s_lets)
plt.imshow(image2)


image3 = shepp_sirt(nPix,dPix,nTheta,d_lets,s_lets)
plt.imshow(image3)

y = np.vstack([image[32,:],image3[32,:]])
vt.CreatePlot(y.T, labels=["True","SIRT"] )

vt.CreateImage(true,outfile='true')
vt.CreateImage(image3, outfile='sirt')
vt.CreateImage((image3-true), ctitle= " ", outfile='diff')
