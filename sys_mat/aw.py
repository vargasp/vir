# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 21:38:08 2026

@author: varga
"""

import numpy as np

# Small epsilon to avoid numerical problems when a ray lies exactly
# on a grid line or boundary.
eps = 1e-6


def _fp_2d_intersect_bounding(r0, dr, adr, r_min, r_max):
    # Intersect ray with image bounding box
    #
    # We compute parametric entry/exit values t0, t1 such that:
    #   r(t0) enters the image
    #   r(t1) exits the image
    
    
    if adr > eps:
        tr0 = (r_min - r0) / dr
        tr1 = (r_max - r0) / dr
        trmin = min(tr0, tr1)
        trmax = max(tr0, tr1)
    else:
        # Ray is (almost) vertical → no x-bound intersection
        trmin = -np.inf
        trmax =  np.inf

    return trmin, trmax


def _fp_2d_step_init(r0, ir, dr, adr, r_min, d_pix):
    # Amanatides–Woo stepping initialization
    #
    # For each axis:
    #   - step_x / step_y indicates which direction we move
    #   - tMaxX / tMaxY is the parametric distance to the next
    #     vertical/horizontal grid boundary
    #   - tDeltaX / tDeltaY is the distance between crossings
    if dr > 0:
        step_r = 1
        r_next = r_min + (ir + 1) * d_pix
    else:
        step_r = -1
        r_next = r_min + ir * d_pix

    if adr > eps:
        tDeltaR = d_pix / adr
        tMaxR = (r_next - r0) / dr
    else:
        tDeltaR = np.inf
        tMaxR = np.inf
        
    return step_r, tDeltaR, tMaxR

def _fp_2d_traverse_grid(img,sino,ia,idt,t0,t1,tMaxX,tMaxY,tDeltaX,tDeltaY,
                         step_x,step_y,ix,iy,nX,nY,d_pix):
    
    # Ensure first crossing occurs after entry point
    tMaxX = max(tMaxX, t0)
    tMaxY = max(tMaxY, t0)

    # Traverse the grid voxel-by-voxel
    # At each step:
    #   - Choose the nearest boundary crossing
    #   - Accumulate voxel value × segment length
    #   - Advance to the next voxel
    t = t0
    acc = 0.0

    while t < t1:

        # Safety check (should rarely trigger)
        if ix < 0 or ix >= nX or iy < 0 or iy >= nY:
            break

        if tMaxX <= tMaxY:
            # Cross a vertical boundary first
            tNext = min(tMaxX, t1)
            acc += img[ix, iy] * (tNext - t)
            t = tNext
            tMaxX += tDeltaX
            ix += step_x
        else:
            # Cross a horizontal boundary first
            tNext = min(tMaxY, t1)
            acc += img[ix, iy] * (tNext - t) 
            t = tNext
            tMaxY += tDeltaY
            iy += step_y

    # Store final line integral
    sino[ia, idt] = acc


def aw_fp_2d(img, angs, n_dets, d_det=1.0, d_pix=1.0):
    """
    2D parallel-beam forward projection using Amanatides–Woo ray traversal.

    Computes the line integrals of a 2D image over a set of parallel-beam
    rays defined by projection angles and detector positions.

    Geometry:
        - Image is defined on a Cartesian grid of size (nX, nY)
        - Pixel size is dPix
        - Coordinate system is centered at (0, 0)
        - For angle θ:
            ray direction = (cos θ, sin θ)
            detector offset = det
            ray origin = (-sin θ * det, cos θ * det)

    The projection is computed as:
        sino[ia, idt] = ∫ img(x, y) ds
    where the integral is approximated by summing voxel values weighted
    by exact intersection lengths.

    Parameters
    ----------
    img : ndarray, shape (nX, nY)
        Input image (voxel values).
    angles : ndarray, shape (nAng,)
        Projection angles in radians.
    n_dets : int
        Number of detector bins.
    d_det : float, optional
        Detector spacing (default: 1.0).
    d_pix : float, optional
        Pixel size (default: 1.0).

    Returns
    -------
    sino : ndarray, shape (nAng, n_dets)
        Sinogram (line integrals).
    """

    nx_pix, ny_pix = img.shape          # number of pixels in x and y

    # Output sinogram: one value per (angle, detector)
    sino = np.zeros((angs.size, n_dets), dtype=np.float32)

    # Define image bounding box in world coordinates
    x0 = -0.5 * nx_pix * d_pix
    xn =  0.5 * nx_pix * d_pix
    y0 = -0.5 * ny_pix * d_pix
    yn =  0.5 * ny_pix * d_pix

    # Precompute ray direction for all angles
    cos_angs = np.cos(angs)
    sin_angs = np.sin(angs)

    # Detector bin centers (detector coordinate u)
    cnt_dets = d_det * (np.arange(n_dets) - n_dets/2 + 0.5)

    # Main loops: angles → detectors → voxel traversal
    for i_ang, (cos_ang,sin_ang) in enumerate(zip(cos_angs,sin_angs)):
        
        #Ray directions
        dx = cos_ang
        dy = sin_ang 
        
        # Absolute values are used for step size calculations
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        for i_det, cnt_det in enumerate(cnt_dets):
            # Each ray is parameterized as:
            # r(t) = (x0_ray, y0_ray) + t * (dx, dy)
            x0_ray = -dy * cnt_det
            y0_ray =  dx * cnt_det

            # Intersect ray with image bounding box
            tx_enter, tx_exit = _fp_2d_intersect_bounding(x0_ray, dx, abs_dx, x0, xn)
            ty_enter, ty_exit = _fp_2d_intersect_bounding(y0_ray, dy, abs_dy, y0, yn)

            # Combined entry/exit interval
            t_enter = max(tx_enter, ty_enter)
            t_exit = min(tx_exit,  ty_exit)

            if t_exit <= t_enter:
                # Ray does not intersect the image
                continue

            # Compute entry point into the image
            x = x0_ray + t_enter * dx
            y = y0_ray + t_enter * dy

            # Clamp slightly inside the image to avoid edge ambiguity
            x = min(max(x, x0 + eps), xn - eps)
            y = min(max(y, y0 + eps), yn - eps)

            # Convert entry point to voxel indices
            ix = int(np.floor((x - x0) / d_pix))
            iy = int(np.floor((y - y0) / d_pix))

            # Amanatides–Woo stepping initialization
            step_x, tDeltaX, tMaxX = _fp_2d_step_init(x0_ray, ix, dx, abs_dx, x0, d_pix)
            step_y, tDeltaY, tMaxY = _fp_2d_step_init(y0_ray, iy, dy, abs_dy, y0, d_pix)
            
            # Traverse the grid voxel-by-voxel
            _fp_2d_traverse_grid(img,sino,i_ang,i_det,t_enter,t_exit,tMaxX,tMaxY,tDeltaX,tDeltaY,
                                     step_x,step_y,ix,iy,nx_pix,ny_pix,d_pix)

    #Returns the sino
    return sino


def aw_bp_2d(sino, Angs, img_shape, d_det=1.0, d_pix=1.0):
    """
    2D parallel-beam backprojection using Amanatides–Woo ray traversal.

    This function is the exact adjoint of `aw_fp_2d`. It distributes
    sinogram values back into the image grid using the same ray geometry
    and path-length weighting.

    For each ray:
        img[ix, iy] += sino[ia, idt] * path_length

    No filtering is applied (this is NOT FBP).

    Parameters
    ----------
    sino : ndarray, shape (nAng, nDets)
        Sinogram data.
    Angs : ndarray, shape (nAng,)
        Projection angles in radians.
    img_shape : tuple of int
        Shape of output image (nX, nY).
    d_det : float, optional
        Detector spacing (default: 1.0).
    d_pix : float, optional
        Pixel size (default: 1.0).

    Returns
    -------
    img_bp : ndarray, shape (nX, nY)
        Backprojected image.
    """

    nX, nY = img_shape
    nAng, n_dets = sino.shape

    img_bp = np.zeros((nX, nY), dtype=np.float32)

    # Image bounds
    Xmin = -0.5 * nX * d_pix
    Xmax =  0.5 * nX * d_pix
    Ymin = -0.5 * nY * d_pix
    Ymax =  0.5 * nY * d_pix

    eps = 1e-6

    cosA = np.cos(Angs)
    sinA = np.sin(Angs)

    det_pos = d_det * (np.arange(n_dets) - 0.5 * n_dets + 0.5)

    for ia in range(nAng):
        dx = cosA[ia]
        dy = sinA[ia]

        adx = abs(dx)
        ady = abs(dy)

        for idt in range(n_dets):

            # Sinogram value for this ray
            val = sino[ia, idt]
            if val == 0.0:
                continue

            det = det_pos[idt]

            x0 = -dy * det
            y0 =  dx * det

            # Bounding box intersection (identical to FP)
            txmin, txmax = _fp_2d_intersect_bounding(x0, dx, adx, Xmin, Xmax)
            tymin, tymax = _fp_2d_intersect_bounding(y0, dy, ady, Ymin, Ymax)

            t0 = max(txmin, tymin)
            t1 = min(txmax, tymax)

            if t1 <= t0:
                continue

            # Entry point and initial voxel
            x = x0 + t0 * dx
            y = y0 + t0 * dy

            x = min(max(x, Xmin + eps), Xmax - eps)
            y = min(max(y, Ymin + eps), Ymax - eps)

            ix = int(np.floor((x - Xmin) / d_pix))
            iy = int(np.floor((y - Ymin) / d_pix))

            # Stepping setup (identical to FP)
            # Amanatides–Woo stepping initialization
            step_x, tDeltaX, tMaxX = _fp_2d_step_init(x0, ix, dx, adx, Xmin, d_pix)
            step_y, tDeltaY, tMaxY = _fp_2d_step_init(y0, iy, dy, ady, Ymin, d_pix)
            

            tMaxX = max(tMaxX, t0)
            tMaxY = max(tMaxY, t0)

            # Traverse voxels and scatter contribution
            t = t0

            while t < t1:
                if ix < 0 or ix >= nX or iy < 0 or iy >= nY:
                    break

                if tMaxX <= tMaxY:
                    tNext = min(tMaxX, t1)
                    img_bp[ix, iy] += val * (tNext - t)
                    t = tNext
                    tMaxX += tDeltaX
                    ix += step_x
                else:
                    tNext = min(tMaxY, t1)
                    img_bp[ix, iy] += val * (tNext - t)
                    t = tNext
                    tMaxY += tDeltaY
                    iy += step_y

    return img_bp


def aw_fp_2d_fan_flat(img, Angs, n_dets, DSO, DSD, d_det=1.0, d_pix=1.0):
    """
    2D flat-panel fan-beam forward projection using Amanatides–Woo traversal.

    Parameters
    ----------
    img : ndarray (nX, nY)
        Input image.
    Angs : ndarray (nAng,)
        Projection angles in radians.
    n_dets : int
        Number of detector elements.
    DSO : float
        Distance from source to isocenter.
    DSD : float
        Distance from source to detector.
    d_det : float
        Detector spacing.
    d_pix : float
        Pixel size.

    Returns
    -------
    sino : ndarray (nAng, nDets)
        Fan-beam sinogram.
    """

    nx, ny = img.shape
    nAng = Angs.size
    sino = np.zeros((nAng, n_dets), dtype=np.float32)

    # Image bounding box
    x0 = -d_pix * nx / 2
    y0 = -d_pix * ny / 2
    x1 =  d_pix * nx / 2
    y1 =  d_pix * ny / 2

    # Detector coordinates (flat panel)
    det_u = d_det * (np.arange(n_dets) - n_dets/2 + 0.5)

    cos_angles = np.cos(Angs)
    sin_angles = np.sin(Angs)

    for ia, (c,s) in enumerate(zip(cos_angles,sin_angles)):
        # Source position
        xs = DSO * c
        ys = DSO * s

        # Detector center
        xd0 = -(DSD - DSO) * c
        yd0 = -(DSD - DSO) * s

        # Detector direction (tangent to rotation)
        dux = -s
        duy = c

        for idt in range(n_dets):

            # Detector point
            xd = xd0 + det_u[idt] * dux
            yd = yd0 + det_u[idt] * duy

            # Ray direction
            dx = xd - xs
            dy = yd - ys

            L = np.sqrt(dx*dx + dy*dy)
            dx /= L
            dy /= L

            adx = abs(dx)
            ady = abs(dy)

            # --- Bounding box intersection ---
            txmin, txmax = _fp_2d_intersect_bounding(xs, dx, adx, x0, x1)
            tymin, tymax = _fp_2d_intersect_bounding(ys, dy, ady, y0, y1)

            t0 = max(txmin, tymin)
            t1 = min(txmax, tymax)

            if t1 <= t0:
                continue

            # Entry point
            x = xs + t0 * dx
            y = ys + t0 * dy

            x = min(max(x, x0 + eps), x1 - eps)
            y = min(max(y, y0 + eps), y1 - eps)

            ix = int(np.floor((x - x0) / d_pix))
            iy = int(np.floor((y - y0) / d_pix))

            # Amanatides–Woo stepping initialization
            step_x, tDeltaX, tMaxX = _fp_2d_step_init(xs, ix, dx, adx, x0, d_pix)
            step_y, tDeltaY, tMaxY = _fp_2d_step_init(ys, iy, dy, ady, y0, d_pix)
            
            _fp_2d_traverse_grid(img,sino,ia,idt,t0,t1,tMaxX,tMaxY,tDeltaX,tDeltaY,
                         step_x,step_y,ix,iy,nx,ny,d_pix)

    return sino


def aw_bp_2d_fan_flat(sino, Angs, img_shape, DSO, DSD, dPix=1.0, d_det=1.0):
    """
    Adjoint backprojection for flat-panel fan-beam geometry.
    """

    nX, nY = img_shape
    nAng, n_dets = sino.shape

    img_bp = np.zeros((nX, nY), dtype=np.float32)

    Xmin = -0.5 * nX * dPix
    Xmax =  0.5 * nX * dPix
    Ymin = -0.5 * nY * dPix
    Ymax =  0.5 * nY * dPix

    eps = 1e-6

    cosA = np.cos(Angs)
    sinA = np.sin(Angs)

    det_u = d_det * (np.arange(n_dets) - 0.5 * n_dets + 0.5)

    for ia in range(nAng):

        c = cosA[ia]
        s = sinA[ia]

        xs = -DSO * s
        ys =  DSO * c

        xd0 = (DSD - DSO) * s
        yd0 = -(DSD - DSO) * c

        dux = c
        duy = s

        for idt in range(n_dets):

            val = sino[ia, idt]
            if val == 0.0:
                continue

            xd = xd0 + det_u[idt] * dux
            yd = yd0 + det_u[idt] * duy

            dx = xd - xs
            dy = yd - ys

            adx = abs(dx)
            ady = abs(dy)

            if adx > eps:
                tx0 = (Xmin - xs) / dx
                tx1 = (Xmax - xs) / dx
                txmin = min(tx0, tx1)
                txmax = max(tx0, tx1)
            else:
                txmin = -np.inf
                txmax =  np.inf

            if ady > eps:
                ty0 = (Ymin - ys) / dy
                ty1 = (Ymax - ys) / dy
                tymin = min(ty0, ty1)
                tymax = max(ty0, ty1)
            else:
                tymin = -np.inf
                tymax =  np.inf

            t0 = max(txmin, tymin)
            t1 = min(txmax, tymax)

            if t1 <= t0:
                continue

            x = xs + t0 * dx
            y = ys + t0 * dy

            x = min(max(x, Xmin + eps), Xmax - eps)
            y = min(max(y, Ymin + eps), Ymax - eps)

            ix = int(np.floor((x - Xmin) / dPix))
            iy = int(np.floor((y - Ymin) / dPix))

            if dx > 0:
                step_x = 1
                x_next = Xmin + (ix + 1) * dPix
            else:
                step_x = -1
                x_next = Xmin + ix * dPix

            if dy > 0:
                step_y = 1
                y_next = Ymin + (iy + 1) * dPix
            else:
                step_y = -1
                y_next = Ymin + iy * dPix

            tDeltaX = dPix / adx if adx > eps else np.inf
            tDeltaY = dPix / ady if ady > eps else np.inf

            tMaxX = (x_next - xs) / dx if adx > eps else np.inf
            tMaxY = (y_next - ys) / dy if ady > eps else np.inf

            tMaxX = max(tMaxX, t0)
            tMaxY = max(tMaxY, t0)

            t = t0

            while t < t1:
                if ix < 0 or ix >= nX or iy < 0 or iy >= nY:
                    break

                if tMaxX <= tMaxY:
                    tNext = min(tMaxX, t1)
                    img_bp[ix, iy] += val * (tNext - t)
                    t = tNext
                    tMaxX += tDeltaX
                    ix += step_x
                else:
                    tNext = min(tMaxY, t1)
                    img_bp[ix, iy] += val * (tNext - t)
                    t = tNext
                    tMaxY += tDeltaY
                    iy += step_y

    return img_bp
