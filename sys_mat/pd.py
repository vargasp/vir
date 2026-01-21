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

    # Image edges (centered)
    x_bnd_arr = d_pix*(np.arange(nx+1, dtype=np.float32) - nx/2)
    y_bnd_arr = d_pix*(np.arange(ny+1, dtype=np.float32) - ny/2)

    # Detector edges:
    u_bnd_arr = du*(np.arange(nu + 1, dtype=np.float32) - nu/2.0 + su)

    # Precompute trig functions for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):
        
        x_bnd_l = x_bnd_arr[0]
        for ix, x_bnd_r in enumerate(x_bnd_arr[1:]):
            p_bnd_l = -sin_ang*x_bnd_l
            p_bnd_r = -sin_ang*x_bnd_r
            
            y_bnd_l = y_bnd_arr[0]                
            for iy, y_bnd_r in enumerate(y_bnd_arr[1:]):
                o_bnd_l = cos_ang*y_bnd_l
                o_bnd_r = cos_ang*y_bnd_r
                

                if img[ix, iy] == 0:
                    y_bnd_l = y_bnd_r
                    continue

                # Project pixel corners to detector axis
                corners = [o_bnd_l + p_bnd_l, o_bnd_r + p_bnd_l,
                           o_bnd_l + p_bnd_r, o_bnd_r + p_bnd_r]

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
    x_bnd_arr = d_pix * (np.arange(nx + 1) - nx / 2)
    y_bnd_arr = d_pix * (np.arange(ny + 1) - ny / 2)

    # Detector bin boundaries (u)
    u_bnd = du * (np.arange(nu + 1) - nu / 2 + su)

    # Precompute trig functions for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)
    
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):
    
        x_bnd_l = x_bnd_arr[0]
        for ix, x_bnd_r in enumerate(x_bnd_arr[1:]):
    
                
            y_bnd_l = y_bnd_arr[0]                
            for iy, y_bnd_r in enumerate(y_bnd_arr[1:]):
    
                val = img[ix, iy]
                if val == 0:
                    y_bnd_l = y_bnd_r
                    continue

                
                denom = DSO - (x_bnd_l * cos_ang + y_bnd_l * sin_ang)
                u1 = DSD * (-x_bnd_l * sin_ang + y_bnd_l * cos_ang) / denom

                denom = DSO - (x_bnd_r * cos_ang + y_bnd_l * sin_ang)
                u2 = DSD * (-x_bnd_r * sin_ang + y_bnd_l * cos_ang) / denom

                denom = DSO - (x_bnd_l * cos_ang + y_bnd_r * sin_ang)
                u3 = DSD * (-x_bnd_l * sin_ang + y_bnd_r * cos_ang) / denom

                denom = DSO - (x_bnd_r * cos_ang + y_bnd_r * sin_ang)
                u4 = DSD * (-x_bnd_r * sin_ang + y_bnd_r * cos_ang) / denom


                umin = min([u1,u2,u3,u4])
                umax = max([u1,u2,u3,u4])


                
                # Overlapping detector bins
                iu0 = np.searchsorted(u_bnd, umin, side="right") - 1
                iu1 = np.searchsorted(u_bnd, umax, side="left")

                for iu in range(max(0, iu0), min(nu, iu1)):
                    left = max(umin, u_bnd[iu])
                    right = min(umax, u_bnd[iu + 1])
                    if right > left:
                        sino[ia, iu] += val * (right - left) / du

                y_bnd_l = y_bnd_r
            x_bnd_l = x_bnd_r

    return sino


