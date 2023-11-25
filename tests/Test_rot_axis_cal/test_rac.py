#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 07:37:42 2023

@author: pvargas21
"""

import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import vir.rot_cal as rc


def discrete_sphere(nPixels=(128,128), center=None, radius=10):
    f = 4
    nPixels = np.array(nPixels)*f
    
    if center is None:
        center = nPixels/2.0
    else:
        center = np.array(center)*f + nPixels/2.0
            
    x,y = np.indices((nPixels[0],nPixels[1])) + 0.5
    
    phantom = np.zeros(nPixels)
    phantom[(x - center[0])**2 + (y - center[1])**2 < (radius*f)**2] = 1.0
    
    sh = int(nPixels[0]/f), phantom.shape[0]//int(nPixels[0]/f),\
         int(nPixels[1]/f), phantom.shape[1]//int(nPixels[1]/f)
    return phantom.reshape(sh).mean(-1).mean(1)


    
def rotateImage(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR =rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]


def simple_forward_pivot(image, angles, pivot):
    nbins = np.min(image.shape)
    nangles = angles.size
    sino = np.zeros([nbins,nangles])
    
    for i, angle in enumerate(angles):
        sino[:, i] = np.sum(rotateImage(image, angle, pivot), axis=0)
    
    return sino    
    
angles = np.linspace(0,360,360, endpoint=False)

sino0 = []
sino1 = []
sino2 = []
sino3 = []
phantom = np.zeros([128,128])
p_radii = (20,5,5,5,5,5,5,5,5,5)
p_centers = ( (0,0), (0,10), (10,0), (-10,0), (0,-10),\
             (10,10), (-10,-10), (-10,10), (10,-10))

    
for p_radius, p_center in zip(p_radii,p_centers):
    phantom += discrete_sphere(center=p_center, radius=p_radius)
    sino0.append(simple_forward_pivot(phantom, angles,(64,64)))                                
    sino1.append(simple_forward_pivot(phantom, angles,(64,69)))                                   
    sino2.append(simple_forward_pivot(phantom, angles,(69,64)))                                 
    sino3.append(simple_forward_pivot(phantom, angles,(69,69)))                                  

                 

cog0 = np.zeros([9,360])
cog1 = np.zeros([9,360])
cog2 = np.zeros([9,360])
cog3 = np.zeros([9,360])

for i in range(9):
    cog0[i,:] = np.sum(sino0[i]*np.arange(1,129)[:,np.newaxis], axis=0)/np.sum(sino0[i],axis=0)
    cog1[i,:] = np.sum(sino1[i]*np.arange(1,129)[:,np.newaxis], axis=0)/np.sum(sino1[i],axis=0)
    cog2[i,:] = np.sum(sino2[i]*np.arange(1,129)[:,np.newaxis], axis=0)/np.sum(sino2[i],axis=0)
    cog3[i,:] = np.sum(sino3[i]*np.arange(1,129)[:,np.newaxis], axis=0)/np.sum(sino3[i],axis=0)



#plt.imshow(phantom, origin='lower')
vol0 = np.stack([sino0_0,sino0_1,sino0_2,sino0_3], axis=1)
vol1 = np.stack([sino1_0,sino1_1,sino1_2,sino1_3], axis=1)
vol2 = np.stack([sino2_0,sino2_1,sino2_2,sino2_3], axis=1)
vol3 = np.stack([sino3_0,sino3_1,sino3_2,sino3_3], axis=1)




import vir.rot_cal as rc
t0 = rc.correct_rotation_axis(vol0, max_error=10)


t0 = rc.correct_rotation_axis(vol0, max_error=10)

t1 = rc.correct_rotation_axis(vol1, max_error=10)

t2 = rc.correct_rotation_axis(vol2, max_error=10)


t3= rc.correct_rotation_axis(vol3, max_error=10)



vol = np.stack([sino_offset,sino_truth], axis=1)
test = rc.correct_rotation_axis(vol, max_error=10)

