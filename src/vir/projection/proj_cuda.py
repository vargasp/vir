# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 10:01:03 2025

@author: varga
"""

import numpy as np

from numba import cuda



@cuda.jit
def proj_cuda(phantom_flat, idxs, lengths, offsets, sino_flat):
    i = cuda.grid(1)  # 1D grid of threads, each thread does a ray
    n_rays = offsets.size - 1
    
    if i < n_rays:
        sIdx = offsets[i]
        eIdx = offsets[i+1]
        s = 0.0
        # Main ray sum loop (can be thousands of segments per ray!)
        for j in range(sIdx, eIdx):
            s += phantom_flat[idxs[j]] * lengths[j]
            
        sino_flat[i] = s  # Output for this ray


def sd_f_proj_numba_g(phantom, sdlist,sino_shape):
    sino = np.zeros(sino_shape, dtype=np.float32)
    
    #Creates views of the data a 1d arrays for faster iteration    
    n_rays = sdlist[2].size - 1  
    sino_flat = sino.ravel()
    phantom_flat = phantom.ravel()
    idxs, lengths, offsets = sdlist

    # ==== Transfer arrays to GPU ====
    phantom_flat_device = cuda.to_device(phantom_flat)
    idxs_device = cuda.to_device(idxs)
    lengths_device = cuda.to_device(lengths)
    offsets_device = cuda.to_device(offsets)
    sino_flat_device = cuda.device_array_like(sino_flat)
    
    # ==== CUDA kernel launch settings ====
    threads_per_block = 128
    blocks = (n_rays + threads_per_block - 1) // threads_per_block
    
    # ==== Launch the kernel ====
    proj_cuda[blocks, threads_per_block](
        phantom_flat_device,
        idxs_device,
        lengths_device,
        offsets_device,
        sino_flat_device
    )
    
    # ==== Copy result back ====
    sino_flat_result = sino_flat_device.copy_to_host()

    return sino_flat_result.reshape(sino_shape)    
