#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 14:03:03 2025

@author: pvargas21
"""

import numpy as np

def siddon_precompute_3d(sources, detectors, volume_shape, voxel_size):
    """
    Precompute voxel indices and exact intersection lengths for all rays using Siddon method.
    
    Returns:
        rays: list of dictionaries with keys:
            'indices': list of (x, y, z) voxel indices
            'lengths': corresponding intersection lengths
    """
    nx, ny, nz = volume_shape
    vx, vy, vz = voxel_size
    rays = []

    for src, det in zip(sources, detectors):
        ray = {'indices': [], 'lengths': []}

        src = np.array(src)
        det = np.array(det)
        d = det - src
        L = np.linalg.norm(d)
        ray_dir = d / L

        # Compute voxel boundaries
        x_boundaries = np.arange(0, nx+1) * vx
        y_boundaries = np.arange(0, ny+1) * vy
        z_boundaries = np.arange(0, nz+1) * vz

        # Calculate t values where ray intersects voxel planes
        tx = (x_boundaries - src[0]) / (ray_dir[0] if ray_dir[0]!=0 else 1e-12)
        ty = (y_boundaries - src[1]) / (ray_dir[1] if ray_dir[1]!=0 else 1e-12)
        tz = (z_boundaries - src[2]) / (ray_dir[2] if ray_dir[2]!=0 else 1e-12)

        t_vals = np.sort(np.concatenate([tx, ty, tz]))
        t_min, t_max = max(t_vals[0], 0), min(t_vals[-1], L)
        if t_max <= t_min:
            rays.append(ray)
            continue

        # Iterate through each voxel along ray
        for i in range(len(t_vals)-1):
            t0, t1 = t_vals[i], t_vals[i+1]
            if t1 <= t_min or t0 >= t_max:
                continue
            t_start = max(t0, t_min)
            t_end = min(t1, t_max)
            if t_end <= t_start:
                continue

            # midpoint to find voxel indices
            mid = src + (t_start + t_end)/2 * ray_dir
            ix, iy, iz = np.floor(mid / voxel_size).astype(int)
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                ray['indices'].append((ix,iy,iz))
                ray['lengths'].append(t_end - t_start)
        rays.append(ray)
    return rays

def forward_projection_siddon(volume, rays):
    """Forward projection using precomputed Siddon voxel lengths."""
    projections = []
    for ray in rays:
        val = sum(volume[ix,iy,iz]*length for (ix,iy,iz), length in zip(ray['indices'], ray['lengths']))
        projections.append(val)
    return np.array(projections)

def back_projection_siddon(volume, rays, projections):
    """Backprojection using precomputed Siddon voxel lengths."""
    for ray, proj_val in zip(rays, projections):
        for (ix,iy,iz), length in zip(ray['indices'], ray['lengths']):
            volume[ix,iy,iz] += proj_val * length
3. Example Usage
python
Copy code
# Volume setup
nx, ny, nz = 64, 64, 64
voxel_size = np.array([1.0, 1.0, 1.0])
volume = np.ones((nx, ny, nz))

# Define rays (source and detector points)
sources = [np.array([-10,32,32]), np.array([-10,0,32])]
detectors = [np.array([70,32,32]), np.array([70,0,32])]

# Precompute Siddon geometry
rays = siddon_precompute_3d(sources, detectors, volume.shape, voxel_size)

# Forward projection
proj = forward_projection_siddon(volume, rays)
print("Forward projection:", proj)

# Backprojection
back_projection_siddon(volume, rays, proj)
print("Sum of volume after backprojection:", np.sum(volume))