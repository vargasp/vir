#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:39:39 2024

@author: pvargas21
"""

import numpy as np
import matplotlib.pyplot as plt

import vir.affine_transforms as af



nX = 128
nY = 128
nZ = 64

phantom2d = np.zeros([nX, nY])
phantom2d[32:96,32:96] = 1
coords = af.coords_array((nX,nY), ones=True)


angles = np.linspace(0,np.pi/2,1000)
angles_m = np.zeros(angles.size)


for i, angle in enumerate(angles):
    R = af.rotateMat((0,angle/100,0), center=(64,0))

    R = np.linalg.inv(R)
    RC = (R @ coords)
    test = af.coords_transform(phantom2d, RC)
    angles_m[i] = np.arccos(np.sum(test, axis=0).max()/2/32)


plt.plot(angles)
plt.plot(angles_m)
plt.show()

