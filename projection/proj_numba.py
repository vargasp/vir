# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 09:55:53 2025

@author: varga
"""
import numpy as np

from numba import njit, prange

@njit
def fast_forward_project(phantom_flat, idxs, lengths, offsets, sino_flat):
    n_rays = offsets.size - 1
    for i in range(n_rays):  
        sIdx = offsets[i]
        eIdx = offsets[i+1]
        s = 0.0
        for j in range(sIdx, eIdx):
            s += phantom_flat[idxs[j]] * lengths[j]
        sino_flat[i] = s


@njit(parallel=True, fastmath=True)
def fast_forward_project_p(phantom_flat, idxs, lengths, offsets, sino_flat):
    n_rays = offsets.size - 1
    for i in prange(n_rays):  # Multi-threading!
        sIdx = offsets[i]
        eIdx = offsets[i+1]
        s = 0.0
        for j in range(sIdx, eIdx):
            s += phantom_flat[idxs[j]] * lengths[j]
        sino_flat[i] = s


def sd_f_proj_numba(phantom, sdlist,sino_shape):
    sino = np.zeros(sino_shape, dtype=np.float32)
    
    #Creates views of the data a 1d arrays for faster iteration    
    sino_flat = sino.ravel()
    phantom_flat = phantom.ravel()

    fast_forward_project(phantom_flat, sdlist[0], sdlist[1], sdlist[2], sino_flat)

    return sino


def sd_f_proj_numba_p(phantom, sdlist,sino_shape):
    sino = np.zeros(sino_shape, dtype=np.float32)
    
    #Creates views of the data a 1d arrays for faster iteration    
    sino_flat = sino.ravel()
    phantom_flat = phantom.ravel()

    fast_forward_project_p(phantom_flat, sdlist[0], sdlist[1], sdlist[2], sino_flat)

    return sino

