# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 21:38:08 2026

@author: varga
"""

import numpy as np

# Small epsilon to avoid numerical problems when a ray lies exactly
# on a grid line or boundary.
eps = 1e-6


"""
def jospeh(det_cnt,ang_sin,ang_cos,d_pix,t0,t1 ):
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
"""

def _fp_2d_intersect_bounding(r0, dr, adr, r_min, r_max):
    # Intersect ray with image bounding box
    #
    # We compute parametric entry/exit values for a single component of r
    #   r(tr_enter) enters the image
    #   r(tr_exit) exits the image
    
    
    if adr > eps:
        tr_enter = (r_min - r0) / dr
        tr_exit = (r_max - r0) / dr
        tr_min = min(tr_enter, tr_exit)
        tr_max = max(tr_enter, tr_exit)
    else:
        # Ray is (almost) vertical → no x-bound intersection
        tr_min = -np.inf
        tr_max =  np.inf

    return tr_min, tr_max


def _fp_2d_step_init(r0, ir, dr, adr, r_min, d_pix):
    # Amanatides–Woo stepping initialization
    #
    # For each axis:
    #   - ix_dir / iy_dir indicates which direction we move
    #   - ix_next / iy_next is the parametric distance to the next
    #     vertical/horizontal grid boundary
    #   - ix_step / iy_step is the distance between crossings
    if dr > 0:
        ir_dir = 1
        r_next = r_min + (ir + 1) * d_pix
    else:
        ir_dir = -1
        r_next = r_min + ir * d_pix

    if adr > eps:
        ir_step = d_pix / adr
        ir_next = (r_next - r0) / dr
    else:
        ir_step = np.inf
        ir_next = np.inf
        
    return ir_dir, ir_step, ir_next


def _fp_2d_traverse_grid(img,sino,ia,iu,t_enter,t_exit,ix_next,iy_next,
                         ix_step,iy_step,ix_dir,iy_dir,ix,iy,nx,ny,d_pix):
    
    # Ensure first crossing occurs after entry point
    ix_next = max(ix_next, t_enter)
    iy_next = max(iy_next, t_enter)

    # Traverse the grid voxel-by-voxel
    # At each step:
    #   - Choose the nearest boundary crossing
    #   - Accumulate voxel value × segment length
    #   - Advance to the next voxel
    t = t_enter
    acc = 0.0

    while t < t_exit:

        # Safety check (should rarely trigger)
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            break

        if ix_next <= iy_next:
            # Cross a vertical boundary first
            t_next = min(ix_next, t_exit)
            acc += img[ix, iy] * (t_next - t)
            t = t_next
            ix_next += ix_step
            ix += ix_dir
        else:
            # Cross a horizontal boundary first
            t_next = min(iy_next, t_exit)
            acc += img[ix, iy] * (t_next - t) 
            t = t_next
            iy_next += iy_step
            iy += iy_dir

    # Store final line integral
    sino[ia, iu] = acc
    

def aw_fp_par_2d(img, ang_arr, nu, du=1.0, su=0.0, d_pix=1.0):
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
    ang_arr : ndarray, shape (nang,)
        Projection angles in radians.
    nu : int
        Number of detector bins.
    du : float, optional
        Detector spacing (default: 1.0).
    d_pix : float, optional
        Pixel size (default: 1.0).

    Returns
    -------
    sino : ndarray, shape (nAng, n_dets)
        Sinogram (line integrals).
    """

    # number of pixels in x and y
    nx, ny = img.shape         

    # Output sinogram: (angle, detector)
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    # Define image bounds in world coordinates
    x_min = -d_pix*nx/2
    x_max =  d_pix*nx/2
    y_min = -d_pix*ny/2
    y_max =  d_pix*ny/2

    # Detector bin centers (detector coordinate u)
    u_arr = du*(np.arange(nu) - nu/2 + 0.5 + su)

    # Precompute ray direction for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    # Main loops: angles → detectors → voxel traversal
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):
        
        #Ray directions
        ray_rx_dir = cos_ang
        ray_ry_dir = sin_ang 
        
        # Absolute values are used for step size calculations
        ray_rx_dir_abs = abs(ray_rx_dir)
        ray_ry_dir_abs = abs(ray_ry_dir)

        for iu, u in enumerate(u_arr):
            # Each ray is parameterized as:
            # r(t) = (ray_o_x_pos, ray_o_y_pos) + t * (ray_r_x_dir, ray_r_y_dir)
            ray_orig_x_pos = -ray_ry_dir * u
            ray_orig_y_pos =  ray_rx_dir * u

            # Intersect ray with image bounding box
            tx_min, tx_max = _fp_2d_intersect_bounding(ray_orig_x_pos, ray_rx_dir, ray_rx_dir_abs, x_min, x_max)
            ty_min, ty_max = _fp_2d_intersect_bounding(ray_orig_y_pos, ray_ry_dir, ray_ry_dir_abs, y_min, y_max)

            # Combined entry/exit interval
            t_enter = max(tx_min, ty_min)
            t_exit = min(tx_max,  ty_max)

            if t_exit <= t_enter:
                # Ray does not intersect the image
                continue

            # Compute entry point into the image
            x_pos = ray_orig_x_pos + t_enter * ray_rx_dir
            y_pos = ray_orig_y_pos + t_enter * ray_ry_dir

            # Clamp slightly inside the image to avoid edge ambiguity
            x_pos = min(max(x_pos, x_min + eps), x_max - eps)
            y_pos = min(max(y_pos, y_min + eps), y_max - eps)

            # Convert entry point to voxel indices
            ix = int(np.floor((x_pos - x_min) / d_pix))
            iy = int(np.floor((y_pos - y_min) / d_pix))

            # Amanatides–Woo stepping initialization
            ix_dir, ix_step, ix_next = _fp_2d_step_init(ray_orig_x_pos, ix, ray_rx_dir, ray_rx_dir_abs, x_min, d_pix)
            iy_dir, iy_step, iy_next = _fp_2d_step_init(ray_orig_y_pos, iy, ray_ry_dir, ray_ry_dir_abs, y_min, d_pix)
            
            # Traverse the grid voxel-by-voxel
            _fp_2d_traverse_grid(img, sino, ia, iu, t_enter, t_exit,
                                 ix_next, iy_next, ix_step, iy_step, ix_dir, iy_dir,
                                 ix, iy, nx, ny, d_pix)

    #Returns the sino
    return sino


def aw_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=1.0, su=0.0, d_pix=1.0):
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
    du : float
        Detector spacing.
    d_pix : float
        Pixel size.

    Returns
    -------
    sino : ndarray (nAng, nDets)
        Fan-beam sinogram.
    """

    # number of pixels in x and y
    nx, ny = img.shape

    # Output sinogram: (angle, detector)
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    # Define image bounds in world coordinates
    x_min = -d_pix * nx / 2
    x_max =  d_pix * nx / 2
    y_min = -d_pix * ny / 2
    y_max =  d_pix * ny / 2

    # Detector bin centers
    u_arr = du*(np.arange(nu) - nu/2 + 0.5 + su)

    # Precompute ray direction for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    # Main loops: angles → detectors → voxel traversal
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):

        # Source position
        src_x_pos = DSO * cos_ang
        src_y_pos = DSO * sin_ang

        # Detector reference point
        u_x_ref = -(DSD - DSO) * cos_ang
        u_y_ref = -(DSD - DSO) * sin_ang

        # Detector direction (tangent to rotation)
        s_rx_dir = -sin_ang
        s_ry_dir = cos_ang

        for iu, u in enumerate(u_arr):
            # Each ray is parameterized as:
            # r(t) = (ray_o_x_pos, ray_o_y_pos) + t * (ray_rx_dir, ray_ry_dir)

            # Detector point
            u_x_pos = u_x_ref + u * s_rx_dir
            u_y_pos = u_y_ref + u * s_ry_dir

            # Ray direction
            ray_rx_dir = u_x_pos - src_x_pos
            ray_ry_dir = u_y_pos - src_y_pos

            ray_len = np.sqrt(ray_rx_dir**2 + ray_ry_dir**2)
            ray_rx_dir /= ray_len
            ray_ry_dir /= ray_len

            # Absolute values are used for step size calculations
            ray_rx_dir_abs = abs(ray_rx_dir)
            ray_ry_dir_abs = abs(ray_ry_dir)

            # Intersection with image bounding box
            t_x_min, t_x_max = _fp_2d_intersect_bounding(src_x_pos, ray_rx_dir, ray_rx_dir_abs, x_min, x_max)
            t_y_min, t_y_max = _fp_2d_intersect_bounding(src_y_pos, ray_ry_dir, ray_ry_dir_abs, y_min, y_max)

            t_enter = max(t_x_min, t_y_min)
            t_exit = min(t_x_max, t_y_max)

            if t_exit <= t_enter:
                continue

            # Entry point (clamped)
            x_pos = src_x_pos + t_enter * ray_rx_dir
            y_pos = src_y_pos + t_enter * ray_ry_dir
            
            x_pos = min(max(x_pos, x_min + eps), x_max - eps)
            y_pos = min(max(y_pos, y_min + eps), y_max - eps)

            # Convert to voxel indices
            ix = int(np.floor((x_pos - x_min) / d_pix))
            iy = int(np.floor((y_pos - y_min) / d_pix))

            # Amanatides–Woo stepping initialization
            ix_dir, ix_step, ix_next = _fp_2d_step_init(src_x_pos, ix, ray_rx_dir, ray_rx_dir_abs, x_min, d_pix)
            iy_dir, iy_step, iy_next = _fp_2d_step_init(src_y_pos, iy, ray_ry_dir, ray_ry_dir_abs, y_min, d_pix)
            
            # Grid traversal
            _fp_2d_traverse_grid(img,sino,ia,iu,t_enter,t_exit,ix_next,iy_next,ix_step,iy_step,
                         ix_dir,iy_dir,ix,iy,nx,ny,d_pix)

    return sino


def aw_bp_2d(sino, Angs, img_shape, d_det=1.0, su=0.0, d_pix=1.0):
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
    x_min = -0.5 * nX * d_pix
    x_max =  0.5 * nX * d_pix
    y_min = -0.5 * nY * d_pix
    y_max =  0.5 * nY * d_pix

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
            txmin, txmax = _fp_2d_intersect_bounding(x0, dx, adx, x_min, x_max)
            tymin, tymax = _fp_2d_intersect_bounding(y0, dy, ady, y_min, y_max)

            t0 = max(txmin, tymin)
            t1 = min(txmax, tymax)

            if t1 <= t0:
                continue

            # Entry point and initial voxel
            x = x0 + t0 * dx
            y = y0 + t0 * dy

            x = min(max(x, x_min + eps), x_max - eps)
            y = min(max(y, y_min + eps), y_max - eps)

            ix = int(np.floor((x - x_min) / d_pix))
            iy = int(np.floor((y - y_min) / d_pix))

            # Stepping setup (identical to FP)
            # Amanatides–Woo stepping initialization
            step_x, tDeltaX, tMaxX = _fp_2d_step_init(x0, ix, dx, adx, x_min, d_pix)
            step_y, tDeltaY, tMaxY = _fp_2d_step_init(y0, iy, dy, ady, y_min, d_pix)
            

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
