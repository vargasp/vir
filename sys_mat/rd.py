# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 21:38:08 2026

@author: varga
"""

import numpy as np

# Small epsilon to avoid numerical problems when a ray lies exactly
# on a grid line or boundary.
eps = 1e-6


def _jospeh(img,d_pix,sino,ia,iu,cos_ang,sin_ang,x_s,y_s,
           x0,y0,t_enter,t_exit,step):

    nx, ny = img.shape

    #Ray directions
    ray_rx_dir = cos_ang
    ray_ry_dir = sin_ang 

    """
    x_min = -d_pix*nx/2
    x_max =  d_pix*nx/2
    y_min = -d_pix*ny/2
    y_max =  d_pix*ny/2
    """

    t = t_enter
    while t <= t_exit:
        # Current position along ray
        x = x_s + ray_rx_dir * t
        y = y_s + ray_ry_dir * t

        """
        # Clamp slightly inside the image to avoid edge ambiguity
        x = min(max(x, x_min + eps), x_max - eps)
        y = min(max(y, y_min + eps), y_max - eps)
        """
        
        # Convert to pixel index
        ix = (x - x0) / d_pix
        iy = (y - y0) / d_pix

        # Bilinear interpolation
        ix = min(max(ix, 0.0), nx - 2.0)
        iy = min(max(iy, 0.0), ny - 2.0)

        ix0 = int(ix)
        iy0 = int(iy)

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
        sino[ia, iu] += val * step
        t += step


def _joseph_bp(img, d_pix, sino, ia, iu, cos_ang, sin_ang, x_s, y_s, x0, y0, t_enter, t_exit, step):
    nx, ny = img.shape
    # Direction components
    ray_rx_dir = cos_ang
    ray_ry_dir = sin_ang

    # Get sinogram value to backproject
    s_val = sino[ia, iu]

    t = t_enter
    while t <= t_exit:
        x = x_s + ray_rx_dir * t
        y = y_s + ray_ry_dir * t

        ix = (x - x0) / d_pix
        iy = (y - y0) / d_pix

        #ix = min(max(ix, 0.0), nx - 2.0)
        #iy = min(max(iy, 0.0), ny - 2.0)
        ix0 = int(ix)
        iy0 = int(iy)

        dx = ix - ix0
        dy = iy - iy0

        # Bilinear splatting
        if ix0>=0 and iy0>=0:
            img[ix0, iy0]       += s_val * (1-dx)*(1-dy) * step

        if ix0+1<nx and iy0>=0:
            img[ix0+1, iy0]     += s_val * dx*(1-dy)     * step

        if ix0>=0 and iy0+1<ny:
            img[ix0, iy0+1]     += s_val * (1-dx)*dy     * step

        if ix0+1<nx and iy0+1<ny:
            img[ix0+1, iy0+1]   += s_val * dx*dy         * step

        t += step


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
    

def _aw_bp_grid(img, sino, ia, iu, t_enter, t_exit,
                ix_next, iy_next, ix_step, iy_step,
                ix_dir, iy_dir, ix, iy, nx, ny, d_pix):
    # Single ray: distribute sinogram to grid voxels
    t = t_enter
    s_val = sino[ia, iu]

    ix_next = max(ix_next, t_enter)
    iy_next = max(iy_next, t_enter)

    while t < t_exit:
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            break
        
        
        if ix_next <= iy_next:
            t_next = min(ix_next, t_exit)
            img[ix, iy] += s_val * (t_next - t)
            t = t_next
            ix_next += ix_step
            ix += ix_dir
        else:
            t_next = min(iy_next, t_exit)
            img[ix, iy] += s_val * (t_next - t)
            t = t_next
            iy_next += iy_step
            iy += iy_dir


def aw_fp_par_2d(img, ang_arr, nu, du=1.0, su=0.0, d_pix=1.0,joseph=False):
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

    #Joseph parameters
    # Grid: centered at zero, units in physical space
    x0 = -d_pix*(nx/2 - 0.5) # First pixel center (x)
    y0 = -d_pix*(ny/2 - 0.5) # First pixel center (y)
    step=.5          


    # Detector bin centers (detector coordinate u)
    u_arr = du*(np.arange(nu) - nu/2 + 0.5 + su)

    # Precompute ray direction for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    # Main loops: angles → detectors → voxel traversal
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):
        
        #Ray directions in unit vector components
        rx = cos_ang
        ry = sin_ang 
        
        # Absolute values are used for step size calculations
        rx_abs = abs(rx)
        ry_abs = abs(ry)

        for iu, u in enumerate(u_arr):
            # Each ray is parameterized as:
            # r(t) = (rx_u, ry_u) + t * (rx, ry)
            
            # Ray origination at the detector
            rx_u = -ry * u
            ry_u =  rx * u

            # Intersect ray with image bounding box
            tx_min, tx_max = _fp_2d_intersect_bounding(rx_u, rx, rx_abs, x_min, x_max)
            ty_min, ty_max = _fp_2d_intersect_bounding(ry_u, ry, ry_abs, y_min, y_max)

            # Combined entry/exit interval
            t_enter = max(tx_min, ty_min)
            t_exit = min(tx_max,  ty_max)

            if t_exit <= t_enter:
                # Ray does not intersect the image
                continue

            if joseph:
            
                _jospeh(img,d_pix,sino,ia,iu,cos_ang,sin_ang,
                       rx_u,ry_u,x0,y0,t_enter,t_exit,step)
            else:
                
                # Compute entry point into the image
                rx_x_min = rx_u + t_enter * rx
                ry_y_min = ry_u + t_enter * ry
    
                # Clamp slightly inside the image to avoid edge ambiguity
                rx_x_min = min(max(rx_x_min, x_min + eps), x_max - eps)
                ry_y_min = min(max(ry_y_min, y_min + eps), y_max - eps)              
                
                # Convert entry point to voxel indices
                ix = int(np.floor((rx_x_min - x_min) / d_pix))
                iy = int(np.floor((ry_y_min - y_min) / d_pix))
    
                # Amanatides–Woo stepping initialization
                ix_dir, ix_step, ix_next = _fp_2d_step_init(rx_u, ix, rx, rx_abs, x_min, d_pix)
                iy_dir, iy_step, iy_next = _fp_2d_step_init(ry_u, iy, ry, ry_abs, y_min, d_pix)
                
                # Traverse the grid voxel-by-voxel
                _fp_2d_traverse_grid(img, sino, ia, iu, t_enter, t_exit,
                                     ix_next, iy_next, ix_step, iy_step, ix_dir, iy_dir,
                                     ix, iy, nx, ny, d_pix)

    #Returns the sino
    return sino


def aw_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=1.0, su=0.0, d_pix=1.0,joseph=False):
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

    #Joseph parameters
    # Grid: centered at zero, units in physical space
    x0 = -d_pix*(nx/2 - 0.5) # First pixel center (x)
    y0 = -d_pix*(ny/2 - 0.5) # First pixel center (y)
    step=.5          



    # Detector bin centers
    u_arr = du*(np.arange(nu) - nu/2 + 0.5 + su)

    # Precompute ray direction for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    # Main loops: angles → detectors → voxel traversal
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):

        # Ray origination at the source
        rx_s = DSO * cos_ang
        ry_s = DSO * sin_ang

        # Detector reference point
        rx_u0 = -(DSD - DSO) * cos_ang
        ry_u0 = -(DSD - DSO) * sin_ang

        # Ray direction (parallel abd orthoganal to detector)
        rp = -sin_ang
        ro = cos_ang

        for iu, u in enumerate(u_arr):
 
            # Each ray is parameterized as:
            # r(t) = (ray_o_x_pos, ray_o_y_pos) + t * (rx, ry)

            # Detector point
            rx_u = rx_u0 + u*rp
            ry_u = ry_u0 + u*ro

            # Ray from source to detector
            rx = rx_u - rx_s
            ry = ry_u - ry_s

            #Ray directions in unit vector components
            ray_len = np.sqrt(rx**2 + ry**2)
            rx /= ray_len
            ry /= ray_len

            # Absolute values are used for step size calculations
            rx_abs = abs(rx)
            ry_abs = abs(ry)

            # Intersection with image bounding box
            t_x_min, t_x_max = _fp_2d_intersect_bounding(rx_s, rx, rx_abs, x_min, x_max)
            t_y_min, t_y_max = _fp_2d_intersect_bounding(ry_s, ry, ry_abs, y_min, y_max)

            t_enter = max(t_x_min, t_y_min)
            t_exit = min(t_x_max, t_y_max)

            if t_exit <= t_enter:
                continue

            if joseph:
    
                _jospeh(img,d_pix,sino,ia,iu,rx,ry,
                       rx_s,ry_s,x0,y0,t_enter,t_exit,step)
            else:

                # Entry point (clamped)
                rx_x_min = rx_s + t_enter * rx
                ry_y_min = ry_s + t_enter * ry
                
                rx_x_min = min(max(rx_x_min, x_min + eps), x_max - eps)
                ry_y_min = min(max(ry_y_min, y_min + eps), y_max - eps)
    
                # Convert to voxel indices
                ix = int(np.floor((rx_x_min - x_min) / d_pix))
                iy = int(np.floor((ry_y_min - y_min) / d_pix))
    
                # Amanatides–Woo stepping initialization
                ix_dir, ix_step, ix_next = _fp_2d_step_init(rx_s, ix, rx, rx_abs, x_min, d_pix)
                iy_dir, iy_step, iy_next = _fp_2d_step_init(ry_s, iy, ry, ry_abs, y_min, d_pix)
                
                # Grid traversal
                _fp_2d_traverse_grid(img,sino,ia,iu,t_enter,t_exit,ix_next,iy_next,ix_step,iy_step,
                             ix_dir,iy_dir,ix,iy,nx,ny,d_pix)

    return sino


def aw_bp_par_2d(sino, ang_arr, img_shape, du=1.0, su=0.0, d_pix=1.0, joseph=False):
    """
    2D parallel-beam ray-driven back-projection.
    Parameters identical to forward AW-projection, but swaps image and sinogram roles.
    """
    nx, ny = img_shape
    nu = sino.shape[1]
    img = np.zeros((nx, ny), dtype=np.float32)

    x_min = -d_pix * nx / 2
    x_max =  d_pix * nx / 2
    y_min = -d_pix * ny / 2
    y_max =  d_pix * ny / 2
    x0 = -d_pix * (nx/2 - 0.5)
    y0 = -d_pix * (ny/2 - 0.5)
    step = .5

    u_arr = du * (np.arange(nu) - nu/2 + 0.5 + su)
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    for ia, (cos_ang, sin_ang) in enumerate(zip(cos_ang_arr, sin_ang_arr)):
        rx = cos_ang
        ry = sin_ang
        rx_abs = abs(rx)
        ry_abs = abs(ry)

        for iu, u in enumerate(u_arr):
            rx_u = -ry * u
            ry_u =  rx * u

            tx_min, tx_max = _fp_2d_intersect_bounding(rx_u, rx, rx_abs, x_min, x_max)
            ty_min, ty_max = _fp_2d_intersect_bounding(ry_u, ry, ry_abs, y_min, y_max)
            t_enter = max(tx_min, ty_min)
            t_exit  = min(tx_max, ty_max)

            if t_exit <= t_enter:
                continue


            if joseph:
                _joseph_bp(img, d_pix, sino, ia, iu, cos_ang, sin_ang, rx_u, ry_u,
                           x0, y0, t_enter, t_exit, step)
            else:
                rx_x_min = rx_u + t_enter * rx
                ry_y_min = ry_u + t_enter * ry
                
                rx_x_min = min(max(rx_x_min, x_min + eps), x_max - eps)
                ry_y_min = min(max(ry_y_min, y_min + eps), y_max - eps)
                
                ix = int(np.floor((rx_x_min - x_min) / d_pix))
                iy = int(np.floor((ry_y_min - y_min) / d_pix))
                
                ix_dir, ix_step, ix_next = _fp_2d_step_init(rx_u, ix, rx, rx_abs, x_min, d_pix)
                iy_dir, iy_step, iy_next = _fp_2d_step_init(ry_u, iy, ry, ry_abs, y_min, d_pix)
                _aw_bp_grid(img, sino, ia, iu, t_enter, t_exit,
                            ix_next, iy_next, ix_step, iy_step, ix_dir, iy_dir,
                            ix, iy, nx, ny, d_pix)
                
                
    return img


def aw_bp_fan_2d(sino, ang_arr, img_shape, DSO, DSD, du=1.0, su=0.0, d_pix=1.0, joseph=False):
    """
    2D fan-beam ray-driven back-projection with flat panel geometry.
    """
    nx, ny = img_shape
    nu = sino.shape[1]
    img = np.zeros((nx, ny), dtype=np.float32)

    x_min = -d_pix * nx / 2
    x_max =  d_pix * nx / 2
    y_min = -d_pix * ny / 2
    y_max =  d_pix * ny / 2
    x0 = -d_pix * (nx/2 - 0.5)
    y0 = -d_pix * (ny/2 - 0.5)
    step = .5

    u_arr = du * (np.arange(nu) - nu/2 + 0.5 + su)
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    for ia, (cos_ang, sin_ang) in enumerate(zip(cos_ang_arr, sin_ang_arr)):
        rx_s = DSO * cos_ang
        ry_s = DSO * sin_ang
        rx_u0 = -(DSD - DSO) * cos_ang
        ry_u0 = -(DSD - DSO) * sin_ang
        rp = -sin_ang
        ro = cos_ang

        for iu, u in enumerate(u_arr):
            rx_u = rx_u0 + u * rp
            ry_u = ry_u0 + u * ro

            rx = rx_u - rx_s
            ry = ry_u - ry_s

            ray_len = np.sqrt(rx ** 2 + ry ** 2)
            rx /= ray_len
            ry /= ray_len
            rx_abs = abs(rx)
            ry_abs = abs(ry)

            t_x_min, t_x_max = _fp_2d_intersect_bounding(rx_s, rx, rx_abs, x_min, x_max)
            t_y_min, t_y_max = _fp_2d_intersect_bounding(ry_s, ry, ry_abs, y_min, y_max)
            t_enter = max(t_x_min, t_y_min)
            t_exit = min(t_x_max, t_y_max)

            if t_exit <= t_enter:
                continue

            if joseph:
                _joseph_bp(img, d_pix, sino, ia, iu, rx, ry, rx_s, ry_s,
                           x0, y0, t_enter, t_exit, step)
            else:
                rx_x_min = rx_s + t_enter * rx
                ry_y_min = ry_s + t_enter * ry
                rx_x_min = min(max(rx_x_min, x_min + eps), x_max - eps)
                ry_y_min = min(max(ry_y_min, y_min + eps), y_max - eps)              

                
                
                ix = int(np.floor((rx_x_min - x_min) / d_pix))
                iy = int(np.floor((ry_y_min - y_min) / d_pix))
                ix_dir, ix_step, ix_next = _fp_2d_step_init(rx_s, ix, rx, rx_abs, x_min, d_pix)
                iy_dir, iy_step, iy_next = _fp_2d_step_init(ry_s, iy, ry, ry_abs, y_min, d_pix)
                _aw_bp_grid(img, sino, ia, iu, t_enter, t_exit,
                            ix_next, iy_next, ix_step, iy_step, ix_dir, iy_dir,
                            ix, iy, nx, ny, d_pix)
    return img



