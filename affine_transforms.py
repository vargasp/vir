# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:30:07 2023

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import transform
from scipy.ndimage import map_coordinates

def rankIdn(A, rank):
    n,m = A.shape
    
    if rank > n or rank >m:
        I = np.identity(rank)
        n,m  = min(n,rank), min(m,rank)
        I[:n,:m] = A[:n,:m]
        return I
    else:
        return A[:rank,:rank]


def transMat(coords,rank=None):
    coords = np.array(coords)
    n = coords.size
    
    if rank == None:
        rank = n+1
    
    T = np.identity(rank)
    T[:n,-1] = coords
    
    return T


def rotateMat(angs, center=None, seq='XYZ', extrinsic=True, rank=2):

    #Converts angs to an np.array and calcualtes the number of angles
    angs = np.array(angs)
    n = angs.size
    
    #If more than one rotation is provided R must be at least rank 3 
    #Rank must be 1 more than the number of translation or have min of 3
    if n > 1: rank = max(rank, 3)    
    if center is None: rank = max(rank, 3)
    if np.array(center).size > 2: rank = max(rank, 4)


    #If the angle is intrinsic lower the sequence 
    if not extrinsic:
        seq = str.lower(seq)
    
    #Match the number of dimensions in the sequence to the number of angles
    seq = seq[(3-n):]
    
    #Calcuatd a 3x3 rotation matrix (centered at the 0,0)
    R = transform.Rotation.from_euler(seq, angs, degrees=True)
    R = R.as_matrix().squeeze()
    R = rankIdn(R, rank)

    #Returns the R matrix or modifies it if rotation center is provided
    if center is None:
        return R
    else:
        T = transMat(center, rank=rank)
        
        return  T @ R @ np.linalg.inv(T)


def coords_array(shape,ones=False):
    if len(shape) == 1:
        coords = np.mgrid[:shape[0]]
    elif len(shape) == 2:
        coords = np.mgrid[:shape[0],:shape[1]]
    elif len(shape) == 3:
        coords = np.mgrid[:shape[0],:shape[1],:shape[2]]
    else:
        print("Dimensions must be between 1 and 3")

    if ones:
        coords = np.concatenate([coords,np.ones(shape)[np.newaxis,...]], axis = 0, dtype=float)
    else:
        coords = np.stack(coords, axis = 0, dtype=float)

    #coords = np.moveaxis(coords,0, -2)
    #coords = np.ascontiguousarray(coords)

    return coords



def coords_transform(arr, coords):
    return map_coordinates(arr, coords, order=1, mode='constant', cval=0.0)




