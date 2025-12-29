#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 13:44:14 2025

@author: pvargas21
"""


import numpy as np


def precompute_joseph_rays_3d(sources, detectors, volume_shape):
    Z, Y, X = volume_shape
    rays = []

    for src, det in zip(sources, detectors):
        x0, y0, z0 = src
        x1, y1, z1 = det

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0

        ray_indices = []
        ray_weights = []

        if abs(dx) >= abs(dy) and abs(dx) >= abs(dz):
            xs = np.arange(np.ceil(min(x0, x1)), np.floor(max(x0, x1)) + 1)
            ts = (xs - x0) / dx
            ys = y0 + ts * dy
            zs = z0 + ts * dz
            ds = np.sqrt(dx*dx + dy*dy + dz*dz) / len(xs)

        elif abs(dy) >= abs(dx) and abs(dy) >= abs(dz):
            ys = np.arange(np.ceil(min(y0, y1)), np.floor(max(y0, y1)) + 1)
            ts = (ys - y0) / dy
            xs = x0 + ts * dx
            zs = z0 + ts * dz
            ds = np.sqrt(dx*dx + dy*dy + dz*dz) / len(ys)

        else:
            zs = np.arange(np.ceil(min(z0, z1)), np.floor(max(z0, z1)) + 1)
            ts = (zs - z0) / dz
            xs = x0 + ts * dx
            ys = y0 + ts * dy
            ds = np.sqrt(dx*dx + dy*dy + dz*dz) / len(zs)

        for x, y, z in zip(xs, ys, zs):
            ix, iy, iz = int(x), int(y), int(z)

            if 0 <= ix < X - 1 and 0 <= iy < Y - 1 and 0 <= iz < Z - 1:
                wx, wy, wz = x - ix, y - iy, z - iz

                for dz_i in (0, 1):
                    for dy_i in (0, 1):
                        for dx_i in (0, 1):
                            w = (
                                (1 - wx if dx_i == 0 else wx) *
                                (1 - wy if dy_i == 0 else wy) *
                                (1 - wz if dz_i == 0 else wz) *
                                ds
                            )
                            ray_indices.append(
                                (iz + dz_i, iy + dy_i, ix + dx_i)
                            )
                            ray_weights.append(w)

        rays.append((np.array(ray_indices), np.array(ray_weights)))

    return rays



def forward_project_precomputed(volume, rays):
    proj = np.zeros(len(rays), dtype=np.float64)

    for i, (indices, weights) in enumerate(rays):
        z, y, x = indices.T
        proj[i] = np.sum(volume[z, y, x] * weights)

    return proj


def backproject_precomputed(projections, rays, volume_shape):
    volume = np.zeros(volume_shape, dtype=np.float64)

    for val, (indices, weights) in zip(projections, rays):
        z, y, x = indices.T
        volume[z, y, x] += val * weights

    return volume