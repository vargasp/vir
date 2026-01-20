# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:27:50 2026

@author: varga
"""

import numpy as np

def pd_fp_par_2d(img, ang_arr, nu, du=1.0, su=0.00, d_pix=1.0):
    """
    Separable footprints forward projector for 2D parallel-beam CT.

    Implements the separable footprint model where each pixel is projected
    as an axis-aligned rectangle (“footprint”) onto the detector array.

    Parameters:
        img     : ndarray shape (nX, nY)
        Angs  : ndarray of projection angles (radians)
        nDets   : number of detector bins
        d_pix    : pixel width
        d_det    : detector bin width

    Returns:
        sino    : ndarray shape (len(angles), nDets)
    """
    nx, ny = img.shape
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    # Detector edges:
    u_bnd_arr = du*(np.arange(nu + 1, dtype=np.float32) - nu/2.0 + su)

    # Image edges (centered)
    x_bnd_arr = d_pix*(np.arange(nx+1, dtype=np.float32) - nx/2)
    y_bnd_arr = d_pix*(np.arange(ny+1, dtype=np.float32) - ny/2)

    # Precompute trig functions for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):
        for ix, (x_min, x_max) in enumerate(zip(x_bnd_arr[:-1], x_bnd_arr[1:])):
            for iy, (y_min, y_max) in enumerate(zip(y_bnd_arr[:-1], y_bnd_arr[1:])):

                if img[ix, iy] == 0:
                    continue

                # Project pixel corners to detector axis
                # These are the min/max of axis_u . (corner position)
                corners = [cos_ang*y_min - sin_ang*x_min, cos_ang*y_max - sin_ang*x_min,
                           cos_ang*y_min - sin_ang*x_max, cos_ang*y_max - sin_ang*x_max]

                P_min = min(corners)
                P_max = max(corners)

                # Footprint: find detector bins that overlap with rectangle projection

                # Bin k: between det_edges[k], det_edges[k+1]
                iu0 = np.searchsorted(u_bnd_arr, P_min, side='right') - 1
                iuN = np.searchsorted(u_bnd_arr, P_max, side='left')

                # For each overlapped bin, calculate geometric overlap
                for iu in range(max(0, iu0), min(nu, iuN)):
                    left = max(P_min, u_bnd_arr[iu])
                    right = min(P_max, u_bnd_arr[iu+1])
                    if right > left:
                        # Normalize by bin width
                        sino[ia, iu] += img[ix, iy] * (right - left) / du

    return sino