#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 15:03:44 2025

@author: pvargas21
"""

import numpy as np

def siddon_joseph_precompute_3d(sources, detectors, volume_shape, voxel_size, n_samples=10):
    """
    Precompute hybrid Siddon-Joseph geometry:
        - Exact voxel intersection lengths (Siddon)
        - Trilinear interpolation weights (Joseph)
    
    Returns a list of rays, each with:
        'indices': list of (x,y,z) voxel indices
        'weights': corresponding weighted contribution per voxel
    """
    nx, ny, nz = volume_shape
    vx, vy, vz = voxel_size
    rays = []

    for src, det in zip(sources, detectors):
        ray = {'indices': [], 'weights': []}
        src = np.array(src, dtype=float)
        det = np.array(det, dtype=float)
        d = det - src
        L = np.linalg.norm(d)
        ray_dir = d / L

        # Sample points along ray (Siddon segments)
        ts = np.linspace(0, L, n_samples)
        segment_length = L / n_samples

        for t in ts:
            pos = src + ray_dir * t
            ix, iy, iz = pos / voxel_size
            ix0, iy0, iz0 = np.floor([ix, iy, iz]).astype(int)
            wx, wy, wz = ix - ix0, iy - iy0, iz - iz0

            for dx in [0,1]:
                for dy in [0,1]:
                    for dz in [0,1]:
                        cx, cy, cz = ix0+dx, iy0+dy, iz0+dz
                        if 0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz:
                            weight = ((1-wx) if dx==0 else wx) * \
                                     ((1-wy) if dy==0 else wy) * \
                                     ((1-wz) if dz==0 else wz)
                            ray['indices'].append((cx,cy,cz))
                            # Multiply by Siddon segment length
                            ray['weights'].append(weight * segment_length)
        rays.append(ray)
    return rays

def forward_projection_hybrid(volume, rays):
    """Forward projection using precomputed Siddon-Joseph rays."""
    projections = []
    for ray in rays:
        val = sum(volume[ix,iy,iz]*w for (ix,iy,iz), w in zip(ray['indices'], ray['weights']))
        projections.append(val)
    return np.array(projections)

def back_projection_hybrid(volume, rays, projections):
    """Backprojection using precomputed Siddon-Joseph rays."""
    for ray, proj_val in zip(rays, projections):
        for (ix,iy,iz), w in zip(ray['indices'], ray['weights']):
            volume[ix,iy,iz] += proj_val * w
            
            
            
# Volume setup
nx, ny, nz = 64, 64, 64
voxel_size = np.array([1.0,1.0,1.0])
volume = np.ones((nx, ny, nz))

# Define rays
sources = [np.array([-10,32,32]), np.array([-10,0,32])]
detectors = [np.array([70,32,32]), np.array([70,0,32])]

# Precompute hybrid Siddon-Joseph rays
rays = siddon_joseph_precompute_3d(sources, detectors, volume.shape, voxel_size, n_samples=20)

# Forward projection
proj = forward_projection_hybrid(volume, rays)
print("Forward projection:", proj)

# Backprojection
back_projection_hybrid(volume, rays, proj)
print("Volume sum after backprojection:", np.sum(volume))