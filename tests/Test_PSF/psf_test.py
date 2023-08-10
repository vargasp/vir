#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:59:05 2023

@author: pvargas21
"""




import numpy as np
import matplotlib.pyplot as plt

import vir
import vir.psf as psf
import vir.fwhm as fwhm


from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import binary_opening
    


N = 40
n = vir.censpace(N)
x2, y2 = np.meshgrid(n,n)
x3, y3, z3 = np.meshgrid(n,n,n)

imp2d = psf.gaussian2d(nX=N,nY=N, mus=(1,-2), sigmas=(3,4),theta=np.pi/4)
params = psf.fitGaussian2d(n, n, imp2d)


imp3d = psf.gaussian3d(nX=N,nY=N,nZ=N, mus=(1,-2,3),sigmas=(5,6,7))
params = psf.fitGaussian3d(n, n, n,imp3d)




cX, cY, cZ = np.unravel_index(np.argmax(img1), img1.shape)

ip = np.where(img1 <=img1.max()*.1, 0, 1)
t = label(ip,connectivity = 1)

imgN = np.where(t==t[cX,cY,cZ],img1,0)




np.where(img1 <=img1.max()*.1, 0, 1)





