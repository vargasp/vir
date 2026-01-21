# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:27:50 2026

@author: varga
"""

import numpy as np

def pd_fp_par_2d(img, ang_arr, nu, du=1.0, su=0.0, d_pix=1.0):
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
        
        x_bnd_l = x_bnd_arr[0]
        for ix, x_bnd_r in enumerate(x_bnd_arr[1:]):
            
            y_bnd_l = y_bnd_arr[0]                 
            for iy, y_bnd_r in enumerate(y_bnd_arr[1:]):

                if img[ix, iy] == 0:
                    y_bnd_l = y_bnd_r
                    continue

                # Project pixel corners to detector axis
                # These are the min/max of axis_u . (corner position)
                #corners = [cos_ang*y_bnd_l - sin_ang*x_bnd_l, cos_ang*y_bnd_r - sin_ang*x_bnd_l,
                #           cos_ang*y_bnd_l - sin_ang*x_bnd_r, cos_ang*y_bnd_r - sin_ang*x_bnd_r]


                c00  = cos_ang*y_bnd_l - sin_ang*x_bnd_l
                c01  = cos_ang*y_bnd_r - sin_ang*x_bnd_l
                c10  = cos_ang*y_bnd_l - sin_ang*x_bnd_r
                c11  = cos_ang*y_bnd_r - sin_ang*x_bnd_r
                 
                corners = [c00, c01, c10, c11]

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

                y_bnd_l = y_bnd_r
            x_bnd_l = x_bnd_r


    return sino




def pd_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=1.0, su=0.0, d_pix=1.0):
    """
    Pixel-driven separable-footprint fan-beam forward projector (2D).
    """
    nx, ny = img.shape
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    # Pixel boundaries (image centered at origin)
    x_bnd = d_pix * (np.arange(nx + 1) - nx / 2)
    y_bnd = d_pix * (np.arange(ny + 1) - ny / 2)

    # Detector bin boundaries (u)
    u_bnd = du * (np.arange(nu + 1) - nu / 2 + su)


    # Precompute trig functions for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)
    

    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):
        for ix in range(nx):
            x0, x1 = x_bnd[ix], x_bnd[ix + 1]

            for iy in range(ny):
                val = img[ix, iy]
                if val == 0:
                    continue

                y0, y1 = y_bnd[iy], y_bnd[iy + 1]

                # Project the four pixel corners
                umin = np.inf
                umax = -np.inf

                for x in (x0, x1):
                    for y in (y0, y1):
                        denom = DSO - (x * cos_ang + y * sin_ang)
                        u = DSD * (-x * sin_ang + y * cos_ang) / denom
                        umin = min(umin, u)
                        umax = max(umax, u)

                # Overlapping detector bins
                iu0 = np.searchsorted(u_bnd, umin, side="right") - 1
                iu1 = np.searchsorted(u_bnd, umax, side="left")

                for iu in range(max(0, iu0), min(nu, iu1)):
                    left = max(umin, u_bnd[iu])
                    right = min(umax, u_bnd[iu + 1])
                    if right > left:
                        sino[ia, iu] += val * (right - left) / du

    return sino


