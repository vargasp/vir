#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 13:13:31 2026

@author: vargasp
"""


import numpy as np

def joseph_fp_2d(img, angles, n_dets, d_det=1.0, d_pix=1.0):
    """
    Joseph's ray-interpolation forward projector for 2D parallel-beam CT.

    Parameters:
        img     : ndarray, shape (nX, nY)
        angles  : ndarray of projection angles [radians]
        nDets   : number of detector bins
        d_pix    : pixel size (width)
        d_det    : detector bin width

    Returns:
        sino    : ndarray (len(angles), nDets)
    """

    nx, ny = img.shape
    sino = np.zeros((angles.size, n_dets), dtype=np.float32)

    # Grid: centered at zero, units in physical space
    x0 = -d_pix * nx/2 + d_pix/2 # First pixel center (x)
    y0 = -d_pix * ny/2 + d_pix/2 # First pixel center (y)

    # Detector bin centers (detector coordinate u)
    u_cnt = d_det*(np.arange(n_dets) - n_dets / 2.0 + 0.5)

    # Project image grid boundaries onto the ray
    # Ray length: covers diagonal of image for safety
    L = d_pix * max(nx, ny) * 2

    # Find t range so that we cover the whole image
    t0 = -L / 2
    t1 = L / 2

    # Precompute trig functions for all angles
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    for i_ang, (ang_cos,ang_sin) in enumerate(zip(cos_angles,sin_angles)):

        # Ray direction is [cos_a, sin_a], detector axis is [-sin_a, cos_a]
        for iDet, det_cnt in enumerate(u_cnt):
            
            # Ray passes through (x_s, y_s)
            x_s = -ang_sin * det_cnt
            y_s = ang_cos * det_cnt

            # Step size along ray (Joseph typically steps in 1-pixel increments)
            step = d_pix / max(abs(ang_cos), abs(ang_sin))

            t = t0
            while t <= t1:
                # Current position along ray
                x = x_s + ang_cos * t
                y = y_s + ang_sin * t

                # Convert to pixel index
                ix = (x - x0) / d_pix
                iy = (y - y0) / d_pix

                if 0 <= ix < nx-1 and 0 <= iy < ny-1:
                    # Bilinear interpolation
                    ix0 = int(np.floor(ix))
                    iy0 = int(np.floor(iy))
                    dx = ix - ix0
                    dy = iy - iy0

                    v00 = img[ix0, iy0]
                    v01 = img[ix0, iy0+1]
                    v10 = img[ix0+1, iy0]
                    v11 = img[ix0+1, iy0+1]

                    val = (v00 * (1-dx)*(1-dy) +
                           v10 * dx * (1-dy) +
                           v01 * (1-dx) * dy +
                           v11 * dx * dy)
                    sino[i_ang, iDet] += val * step
                t += step
    return sino


def joseph_fp_fan_2d(img, Angs, n_dets, DSO, DSD, d_det=1.0, d_pix=1.0):
    """
    Joseph's method forward projector for 2D fan-beam CT.

    Parameters
    ----------
    img : ndarray, shape (nX, nY)
        2D image to project.
    Angs : ndarray, shape (n_angles,)
        Projection angles in radians.
    n_dets : int
        Number of detector bins.
    DSO : float
        Source-to-origin distance.
    DSD : float
        Source-to-detector distance.
    d_det : float
        Detector bin width.
    d_pix : float
        Pixel width.

    Returns
    -------
    sino : ndarray, shape (len(Angs), n_dets)
        Fan-beam sinogram.
    """
    
    

    img = np.ascontiguousarray(img, dtype=np.float32)
    nx, ny = img.shape
    sino = np.zeros((len(Angs), n_dets), dtype=np.float32)

    # Image bounds (physical coordinates)
    x0 = -d_pix * nx / 2
    y0 = -d_pix * ny / 2
    x1 =  d_pix * nx / 2
    y1 =  d_pix * ny / 2

    # Detector coordinates (flat panel)
    det_cnt = d_det * (np.arange(n_dets) - n_dets / 2 + 0.5)

    cos_angles = np.cos(Angs)
    sin_angles = np.sin(Angs)

    for ia, (cos_t,sin_t) in enumerate(zip(cos_angles,sin_angles)):

        # Source position
        src_x = -DSO * sin_t
        src_y =  DSO * cos_t

        # Detector center position (opposite side of origin)
        det_ctr_x =  (DSD - DSO) * sin_t
        det_ctr_y = -(DSD - DSO) * cos_t

        for iDet, u in enumerate(det_cnt):

            # Flat detector pixel position
            det_x = det_ctr_x + u * cos_t
            det_y = det_ctr_y + u * sin_t

            # Ray direction
            dx = det_x - src_x
            dy = det_y - src_y
            norm = np.hypot(dx, dy)
            dx /= norm
            dy /= norm

            # --- Ray–box intersection ---
            if abs(dx) < 1e-12:
                tx_min, tx_max = -np.inf, np.inf
            else:
                tx1 = (x0 - src_x) / dx
                tx2 = (x1 - src_x) / dx
                tx_min, tx_max = min(tx1, tx2), max(tx1, tx2)

            if abs(dy) < 1e-12:
                ty_min, ty_max = -np.inf, np.inf
            else:
                ty1 = (y0 - src_y) / dy
                ty2 = (y1 - src_y) / dy
                ty_min, ty_max = min(ty1, ty2), max(ty1, ty2)

            t_min = max(tx_min, ty_min)
            t_max = min(tx_max, ty_max)

            if t_max <= t_min:
                continue  # Ray misses image entirely

            # Joseph step size
            step = d_pix / max(abs(dx), abs(dy))

            # March only inside image
            t = t_min
            while t <= t_max:
                x = src_x + dx * t
                y = src_y + dy * t

                ix = (x - x0) / d_pix
                iy = (y - y0) / d_pix

                if 0 <= ix < nx-1 and 0 <= iy < ny-1:
                    ix0 = int(ix)
                    iy0 = int(iy)

                    fx = ix - ix0
                    fy = iy - iy0

                    v00 = img[ix0,     iy0]
                    v10 = img[ix0 + 1, iy0]
                    v01 = img[ix0,     iy0 + 1]
                    v11 = img[ix0 + 1, iy0 + 1]

                    val = (
                        v00 * (1 - fx) * (1 - fy) +
                        v10 * fx       * (1 - fy) +
                        v01 * (1 - fx) * fy +
                        v11 * fx       * fy
                    )

                    sino[ia, iDet] += val * step

                t += step

    return sino