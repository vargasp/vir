#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:04:46 2022

@author: vargasp
"""
import os
import ctypes
import time

import numpy as np
import matplotlib.pyplot as plt

import vir
import siddons as sd
import proj_c_test as proj



nDets = 100
nViews = 200
dDet = .25

nPixels = (100,100,1)
dPixel = 1.0

Dets = vir.censpace(nDets, dDet)
Views = np.linspace(0,2*np.pi,nViews, endpoint=False)
trg, src = sd.circular_geom_st(Dets, Views, 500)
sdlist = sd.siddons(trg,src,nPixels,dPixel,ravel=False)


sdlist_c = sd.list_c_object(sdlist)
#del sdlist

phantom = np.zeros(nPixels)
phantom[5,5,0] = 1.0


sino1 = proj.sd_f_proj(phantom, sdlist)
sino2 = proj.sd_f_proj_c(phantom, sdlist)
print(np.max(sino1-sino2))


i1 =  proj.sd_b_proj(sino1, sdlist, nPixels)
i2 =  proj.sd_b_proj_c(sino2, sdlist, nPixels)
print(np.max(i1-i2))
plt.imshow(i2)






sino = np.zeros(sdlist_c.shape + (100,100,))
pt = proj.sd_f_proj_t_c(phantom, sino, sdlist_c, 100, 100)



start = time.time()
p = proj.sd_f_proj_t_c(phantom, sino, sdlist_c, 100, 100)
end = time.time()
print(end - start,(pt - p).max())


phantom = np.zeros((100,100,1))
i = proj.sd_b_proj_t_c(phantom, pt, sdlist_c, 100, 100)

plt.imshow(i)


