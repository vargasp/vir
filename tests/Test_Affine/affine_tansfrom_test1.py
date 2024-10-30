#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:39:39 2024

@author: pvargas21
"""

import numpy as np
import matplotlib.pyplot as plt

import vir.affine_transforms as af

def scaleMat(coords,rank=None):
    """
    Creates an affine translation matrix for translation

    Parameters
    ----------
    coords : float or arraylike
        The translation vector 
    rank : int, optional
        The rank of the matrix. The default is "None" which is the lowest rank

    Returns
    -------
    T : (rank, rank) np.array
        The translation matrix in format: for coords = x; (x,y); and (x,y,z)
        |1 x| if x   |1 0 x| if (x,y)   |1 0 0 x| if (x,y,z)
        |0 1|        |0 1 y|            |0 1 0 y|
                     |0 0 1|            |0 0 1 z|
                                        |0 0 0 1|
    """
    #Calculates the dimension of the coords vector
    coords = np.array(coords)
    coords = np.append(coords,0)
    
    n = coords.size
    
    #Determines the rank of the matrix
    if rank is None:
        rank = n
    
    #Creates the tranlation matrix
    return np.diag(coords)


nX = 128
nY = 128
nZ = 64

phantom2d = np.zeros([nX, nY])
phantom2d[32:96,32:96] = 1





coords = af.coords_array((nX,nY), ones=True)


angles = np.linspace(0,np.pi/2,1000)
angles_m = np.zeros(angles.size)


for i, angle in enumerate(angles):
    R = af.rotateMat((0,angle/100,0), center=(120,0))

    R = np.linalg.inv(R)
    RC = (R @ coords)
    test = af.coords_transform(phantom2d, RC)
    angles_m[i] = np.arccos(np.sum(test, axis=0).max()/2/32)


plt.plot(angles)
plt.plot(angles_m)
plt.show()

