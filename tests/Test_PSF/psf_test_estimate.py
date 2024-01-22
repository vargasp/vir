#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 08:39:29 2024

@author: pvargas21
"""




import numpy as np
import matplotlib.pyplot as plt


import vir
from vir.psf import gaussian2d, estimatePSF
from scipy.signal import convolve2d


psf = gaussian2d(sigmas=(3.0,3.0), nX=21, nY=21)

im_t = np.zeros([129,129])
im_t[50:77,50:74] =1.0
im_b = convolve2d(im_t, psf, mode='same')


mask = 20
kernel_size = psf.shape
psf_e = estimatePSF(im_t, im_b, mask=mask, kernel_size=kernel_size)
plt.imshow(psf_e)
plt.plot(np.array((psf[:,10],psf_e[:,10])).T)

