#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:48:55 2023

@author: pvargas21
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600


from scipy.signal import medfilt
from scipy.ndimage import median_filter



def polar_filter(img, kernel_max, axis,steps=3):

    kernels = np.rint(kernel_max*np.arange(1,steps+1)/steps).astype(int)
    idxs = np.rint(img.shape[axis]*np.arange(0,steps+1)/steps).astype(int)

    lf_img = np.zeros(img.shape)    
    if axis == 0:
        kernels = list(zip(kernels,np.ones(steps, dtype=int)))

        for i, kernel in enumerate(kernels):            
            lf_img[idxs[i]:idxs[i+1],:] = median_filter(img, kernel)[idxs[i]:idxs[i+1],:]

    if axis == 1:
        kernels = list(zip(np.ones(steps, dtype=int),kernels))
        
        for i, kernel in enumerate(kernels):            
            lf_img[:, idxs[i]:idxs[i+1]] = median_filter(img, kernel)[:, idxs[i]:idxs[i+1]]
    
        
    return lf_img



img = np.zeros([300,400])


for i, y in enumerate(np.arange(50,400,50)):
    img[(44-i):(55+i),y] = 1.0
    img[(94-i):(106+i),y] = 1.0
    img[(143-i):(157+i),y] = 1.0
    img[(192-i):(208+i),y] = 1.0
    img[(241-i):(259+i),y] = 1.0

    img[25,(y-5-i):(y+5+i)] = 1.0
    img[75,(y-6-i):(y+6+i)] = 1.0
    img[125,(y-7-i):(y+7+i)] = 1.0
    img[175,(y-8-i):(y+8+i)] = 1.0
    img[225,(y-9-i):(y+9+i)] = 1.0
    img[275,(y-10-i):(y+10+i)] = 1.0



plt.imshow(img.T, origin='lower')



mimg = median_filter(img,size=(1,42))

plt.imshow(mimg.T, origin='lower')




