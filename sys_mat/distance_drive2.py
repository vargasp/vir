# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 21:12:01 2025

@author: varga
"""

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def distance_driven_fp_numba_simd(
    image,
    angles,
    det_count,
    pixel_size=1.0,
    det_spacing=1.0
):
    ny, nx = image.shape
    n_angles = len(angles)

    sinogram = np.zeros((n_angles, det_count), dtype=np.float32)

    det_edges = (np.arange(det_count + 1) - det_count / 2) * det_spacing
    x_edges = (np.arange(nx + 1) - nx / 2) * pixel_size
    y_edges = (np.arange(ny + 1) - ny / 2) * pixel_size

    for ia in prange(n_angles):
        theta = angles[ia]
        c = np.cos(theta)
        s = np.sin(theta)

        if abs(c) >= abs(s):
            # x-driven
            for ix in range(nx):
                col_sum = 0.0
                for iy in range(ny):
                    col_sum += image[iy, ix]

                s0 = x_edges[ix] * c
                s1 = x_edges[ix + 1] * c
                s_min = s0 if s0 < s1 else s1
                s_max = s1 if s1 > s0 else s0

                for idet in range(det_count):
                    overlap = (
                        min(s_max, det_edges[idet + 1])
                        - max(s_min, det_edges[idet])
                    )

                    # branchless accumulation
                    sinogram[ia, idet] += max(overlap, 0.0) * col_sum

        else:
            # y-driven
            for iy in range(ny):
                row_sum = 0.0
                for ix in range(nx):
                    row_sum += image[iy, ix]

                s0 = y_edges[iy] * s
                s1 = y_edges[iy + 1] * s
                s_min = s0 if s0 < s1 else s1
                s_max = s1 if s1 > s0 else s0

                for idet in range(det_count):
                    overlap = (
                        min(s_max, det_edges[idet + 1])
                        - max(s_min, det_edges[idet])
                    )

                    sinogram[ia, idet] += max(overlap, 0.0) * row_sum

    return sinogram