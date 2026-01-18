#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 06:34:23 2026

@author: pvargas21
"""

import numpy as np

def _dd_fp_sweep(sino,img_trm,p_bnd_arr,o_arr,u_bnd_arr,rays_scl_arr,ia):
    """
    Distance-driven sweep kernel for 2D parallel or fanbeam forward projection.

    This function performs the core distance-driven accumulation for a single
    projection angle. It assumes that pixel boundaries projected onto the
    detector coordinate are monotonic along the driving axis, enabling a
    linear-time sweep over pixels and detector bins.

    Geometry:
        - Parallel-beam or fanbeam
        - Detector coordinate u is linear and orthogonal to ray direction

    Parameters
    ----------
    sino : ndarray, shape (n_angles, nu)
        Output sinogram array to be accumulated in-place.
    img_trm : ndarray, shape (np, no)
        Image data after transpose and/or flip so that axis 0 corresponds
        to the driving (sweep) axis.
    p_bnd_arr : ndarray, shape (np + 1,)
        Projected pixel boundary coordinates along the driving axis in
        detector space, independent of orthogonal pixel index.
    o_arr : ndarray, shape (no,)
        Orthogonal pixel-center offsets projected onto the detector coordinate.
    u_bnd_arr : ndarray, shape (nu + 1,)
        Detector bin boundary coordinates in detector space.
    rays_scl_arr : ndarray, shape (np)
        Projection scaling factor (typically 1/|cos(theta)| or 1/|sin(theta)|)
        to  account for pixel width along the ray direction. Fanbean will also
        include a magificaiton factor. The value is inverted to reduce division
        in the hot loop of the sweep function. 
    ia : int
        Index of the current projection angle in the sinogram.

    Notes
    -----
    - The sweep assumes strictly monotonic projected pixel boundaries.
    - Contributions outside the detector range are implicitly discarded.
    - This function does not allocate memory and updates `sino` in-place.
    """
    
    np, no = img_trm.shape
    nu = u_bnd_arr.size - 1

    # Loop over orthogonal pixel lines
    for io in range(no):

        #Hoisting pointers for explicit LICM
        o = o_arr[io]

        img_vec = img_trm[:, io]
        ray_scl = rays_scl_arr[io]

        ip = 0
        iu = 0
        while ip < np and iu < nu:
            
            # Left edge of overlap interval
            # Actual projected pixel boundaries for this pixel row/column
            p_bnd_l = p_bnd_arr[ip] + o
            u_bnd_l = u_bnd_arr[iu]
            overlap_l  = p_bnd_l if p_bnd_l > u_bnd_l else u_bnd_l

            # Right edge of overlap interval
            # Actual projected pixel boundaries for this pixel row/column
            p_bnd_r = p_bnd_arr[ip + 1] + o
            u_bnd_r = u_bnd_arr[iu + 1]
            overlap_r  = p_bnd_r if p_bnd_r < u_bnd_r else u_bnd_r
        
            # Accumulate overlap contribution
            if overlap_r > overlap_l:
                sino[ia, iu] += (img_vec[ip]* (overlap_r - overlap_l)*ray_scl)

            # Advance whichever interval ends first
            if p_bnd_r < u_bnd_r:
                ip += 1
            else:
                iu += 1

def _dd_proj_geom(img_x, img_y, x_bnd_arr, y_bnd_arr, x_arr, y_arr,
                  cos_ang, sin_ang, is_fan=False, DSO=1.0, DSD=1.0):
    
    """
    Prepare distance-driven projection geometry for a single projection angle.

    This function selects a numerically well-conditioned driving axis (x or y),
    projects pixel boundaries onto the detector coordinate, and returns a
    pre-arranged image view such that axis 0 corresponds to the driving
    (sweep) axis required by the distance-driven kernel.

    Two image layouts are accepted:
        - `img_x`: image stored in canonical (x-driven) layout
        - `img_y`: pre-transposed image (y-driven layout)

    This avoids performing a transpose inside the projection loop and ensures
    cache-friendly, stride-1 access in the hot sweep kernel.

    The function supports both parallel-beam and fan-beam geometries and
    guarantees monotonic projected pixel boundaries on return.

    Geometry conventions
    --------------------
    - The detector coordinate `u` is linear and centered at zero.
    - Projection angle is defined by (cos(theta), sin(theta)) = (c_ang, s_ang).
    - Pixel grids are centered at the image origin.
    - Pixel boundary projections along the driving axis are monotonic after
      an optional flip, which is required by the sweep kernel.

    Driving axis selection
    ----------------------
    The driving (sweep) axis is chosen to maximize numerical conditioning:
        - X-driven if |sin(theta)| >= |cos(theta)|
        - Y-driven otherwise

    Fan-beam handling
    -----------------
    When `is_fan=True`, a per-orthogonal-pixel magnification correction is
    applied to approximate true ray path lengths. The correction is inverted
    here so that the sweep kernel uses multiplication instead of division in
    its hot loop.


    Parameters
    ----------
    img_x : ndarray
        Input image in canonical orientation, used when the projection is
        X-driven (axis 0 corresponds to X).
    img_y : ndarray
        Pre-transposed view of the input image, used when the projection is
        Y-driven (axis 0 corresponds to Y).
    x_bnd_arr, y_bnd_arr : ndarray
        Pixel boundary coordinates along X and Y, respectively.
        Shape is (n + 1,) for n pixels.
    x_arr, y_arr : ndarray
        Pixel center coordinates along X and Y, respectively.
        Shape is (n,).
    cos_ang, sin_ang : float
        Cosine and sine of the projection angle.
    is_fan : bool, optional
        If True, apply fan-beam magnification correction. Default is False.
    DSO : float, optional
        Source-to-origin distance (fan-beam only).
    DSD : float, optional
        Source-to-detector distance (fan-beam only).

    Returns
    -------
    img_trm : ndarray
        Image view selected from `img_c` or `img_f`, possibly flipped so that
        axis 0 corresponds to the driving axis and pixel boundaries are
        monotonic in detector space.
    p_bnd_arr : ndarray
        Projected pixel boundary coordinates along the driving axis in
        detector space. Shape is (np + 1,).
    o_arr : ndarray
        Orthogonal pixel-center offsets projected onto the detector
        coordinate. Shape is (no,).
    rays_scale : ndarray
        Per-orthogonal-pixel scaling factors applied in the sweep kernel.
        For parallel-beam geometry this is constant; for fan-beam geometry
        it includes magnification correction.

    Notes
    -----
    - This function performs no allocations in the hot sweep loop.
    - Returned arrays are views whenever possible.
    - Monotonicity of `u_pixs_drive_bnd` is guaranteed on return.
    - No angular fan-beam weighting is applied here; only geometric scaling
      along the detector coordinate is handled.
    """
    
    # Determine driving axis
    if abs(sin_ang) >= abs(cos_ang):
        # X-driven
        
        # Scales projected pixel overlap along X for oblique rays
        ray_scale = abs(sin_ang)

        #Rotate x,y in o,p coordiante system where p is parallel to the detector
        #and o is orthogonal

        # Project pixel boundaries along driving axis onto detector
        # Each X pixel boundary is projected along the ray direction
        p_bnd_arr = -sin_ang * x_bnd_arr

        # Orthogonal pixel-center offset added to base detector projection
        o_arr = cos_ang * y_arr

        #No transformation need  
        img_trm = img_x

    else:
        # Y-driven
        
        # Scales projected pixel overlap along Y for oblique rays
        ray_scale = abs(cos_ang)

        # Project pixel boundaries along driving axis onto detector
        p_bnd_arr = cos_ang * y_bnd_arr

        # Orthogonal pixel-center offset added to base detector projection
        o_arr = -sin_ang * x_arr

        # Transposes image so axis 0 correspondsto the driving (sweep) axis
        img_trm = img_y 

    # Ensure monotonic increasing for sweep
    if p_bnd_arr[1] < p_bnd_arr[0]:
        p_bnd_arr = p_bnd_arr[::-1]
        img_trm = img_trm[::-1, :]

    # Fan-beam magnification correction.
    # Values are inverted to allow multiplicaion in hot loop
    if is_fan:
        # Fan-beam magnification correction:
        # Each pixel's overlap is divided by proj_scale
        # to approximate true line integral along the ray.
        # Multiply by ray_scale to convert driving-axis projection
        # to ray length        
        magnification = DSD / (DSO - o_arr)
        rays_scale = 1/(magnification * ray_scale)
    else:
        #Mirroring fanbeam structure
        rays_scale = np.full(o_arr.size, 1.0/ray_scale, dtype=np.float32)


    return img_trm, p_bnd_arr, o_arr, rays_scale


def dd_fp_par_2d(img, ang_arr, nu, du=1.0, d_pix=1.0):
    """
    Distance-driven forward projection for 2D parallel-beam CT.

    This function computes a sinogram from a 2D image using the
    distance-driven method. For each projection angle, a driving axis
    (x or y) is selected to ensure well-conditioned, monotonic projection
    of pixel boundaries onto the detector.

    Geometry:
        - Parallel-beam
        - Detector centered at origin
        - Rays are orthogonal to detector

    Parameters
    ----------
    img : ndarray, shape (n_x, n_y)
        Input 2D image (attenuation coefficients).
    ang_arr : ndarray, shape (n_angles,)
        Projection angles in radians.
    nu : int
        Number of detector bins.
    d_det : float, optional
        Detector bin width (default is 1.0).
    d_pix : float, optional
        Pixel width (default is 1.0).

    Returns
    -------
    sino : ndarray, shape (n_angles, n_det)
        Computed sinogram.

    Notes
    -----
    - The distance-driven method conserves total image mass under projection.
    - Detector and image grids are assumed to be centered at zero.
    - This implementation performs no explicit ray truncation checks.
    """

    nx, ny = img.shape
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    #Creates two images where the driving axis is contigious
    img_x = np.ascontiguousarray(img)
    img_y = np.ascontiguousarray(img.T)

    # Define pixel boundaries and centers in image space
    # Centered at origin (0,0)
    # x_pixs_bnd, y_pixs_bnd: positions of pixel edges along X and Y
    x_bnd_arr = d_pix*(np.arange(nx+1, dtype=np.float32) - nx/2)
    y_bnd_arr = d_pix*(np.arange(ny+1, dtype=np.float32) - ny/2)
 
    # x_pixs_cnt, y_pixs_cnt: positions of pixel centers
    x_arr = (x_bnd_arr[:-1] + x_bnd_arr[1:])/2
    y_arr = (y_bnd_arr[:-1] + y_bnd_arr[1:])/2

    # Detector bin boundaries along the fan-beam arc
    # Centered at u = 0
    u_bnd_arr = du * (np.arange(nu + 1, dtype=np.float32) - nu / 2.0)

    # Precompute trig functions for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    #Loop through projection angles
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):
       
        img_trm, p_bnd_arr, o_arr, rays_scale = \
            _dd_proj_geom(img_x, img_y, x_bnd_arr, y_bnd_arr, x_arr, y_arr,
                          cos_ang, sin_ang, is_fan=False)
    
        _dd_fp_sweep(sino, img_trm, p_bnd_arr, o_arr,
                         u_bnd_arr, rays_scale, ia)

    #Return sino normalized with pixel and detector size
    return sino*d_pix/du


def dd_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=1.0, d_pix=1.0):
    """
    Distance-driven fan-beam forward projection for 2D CT.

    Computes the sinogram of a 2D image for a set of projection angles
    in fan-beam geometry.

    Parameters
    ----------
    img : ndarray, shape (nX, nY)
        2D image to project.
    ang_arr : ndarray, shape (n_angles,)
        Projection angles in radians.
    nu : int
        Number of detector bins.
    DSO : float
        Source-to-origin distance.
    DSD : float
        Source-to-detector distance.
    dDet : float, optional
        Detector bin width. Default is 1.0.
    dPix : float, optional
        Pixel width in image units. Default is 1.0.

    Returns
    -------
    sino : ndarray, shape (len(Angs), nDets)
        Fan-beam sinogram of the input image.

    Notes
    -----
    - Uses distance-driven accumulation along the driving axis.
    - Corrects for fan-beam magnification to approximate intensity conservation.
    - At oblique angles or for coarse pixels, the distance-driven approximation
      underestimates line integrals along rays (e.g., 45° diagonal rays).
    - For better accuracy:
        1. Use subpixel splitting (divide each pixel into smaller subpixels).
        2. Increase image resolution.
        3. Use ray-driven projection for exact results.
    - Parallel-beam geometry is recovered in the limit DSO → ∞.
    """
    
    nx, ny = img.shape
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    #Creates two images where the driving axis is contigious
    img_x = np.ascontiguousarray(img)
    img_y = np.ascontiguousarray(img.T)


    # Define pixel boundaries and centers in image space
    # Centered at origin (0,0)
    # x_pixs_bnd, y_pixs_bnd: positions of pixel edges along X and Y
    x_bnd_arr = d_pix*(np.arange(nx+1, dtype=np.float32) - nx/2)
    y_bnd_arr = d_pix*(np.arange(ny+1, dtype=np.float32) - ny/2)
 
    # x_pixs_cnt, y_pixs_cnt: positions of pixel centers
    x_arr = (x_bnd_arr[:-1] + x_bnd_arr[1:])/2
    y_arr = (y_bnd_arr[:-1] + y_bnd_arr[1:])/2

    # Detector bin boundaries along the fan-beam arc
    # Centered at u = 0
    u_bnd_arr = du * (np.arange(nu + 1, dtype=np.float32) - nu / 2.0)

    # Precompute trig functions for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    #Loop through projection angles
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):

        img_trm, p_bnd_arr, o_arr, rays_scale = \
           _dd_proj_geom(img_x, img_y, x_bnd_arr, y_bnd_arr, x_arr, y_arr,
                         cos_ang, sin_ang, is_fan=True, DSO=DSO, DSD=DSD)

        # Sweep along driving axis
        _dd_fp_sweep(sino, img_trm, p_bnd_arr, o_arr,
                            u_bnd_arr, rays_scale, ia)

    #Return sino normalized with pixel and detector size
    return sino*d_pix/du




def _dd_bp_par_2d_sweep(img_rot,proj,u_pix_bnd_drive,u_pix_offset_orth,u_det_bnd,n_pix_drive,n_pix_orth,inv):
    """
    Distance-driven sweep for parallel-beam back-projection along one angle.

    Accumulates projection contributions into image pixels along the driving axis.
    This function computes the overlap between projected pixel boundaries and detector
    bins, weighting contributions proportionally to the length of overlap.

    Parameters
    ----------
    img_rot : ndarray, shape (n_pix_drive, n_pix_orth)
        Subarray of the image corresponding to the current driving axis (may be a view).
    proj : ndarray, shape (n_dets,)
        Projection values for the current angle.
    u_pix_bnd_drive : ndarray, shape (n_pix_drive + 1,)
        Pixel boundaries along the driving axis, projected onto the detector coordinate.
    u_pix_offset_orth : ndarray, shape (n_pix_orth,)
        Offsets along the orthogonal axis for each pixel row/column.
    u_det_bnd : ndarray, shape (n_dets + 1,)
        Detector bin boundaries.
    n_pix_drive : int
        Number of pixels along the driving axis.
    n_pix_orth : int
        Number of pixels along the orthogonal axis.
    inv : float
        Scaling factor for projection contribution (typically d_det * |cos(theta)| or |sin(theta)|).

    Notes
    -----
    - This implements the classic distance-driven back-projection for parallel beams.
    - Pixels fully outside the detector range are skipped for efficiency.
    """
    
    n_dets = u_det_bnd.size - 1

    for i_orth in range(n_pix_orth):
        P_bnd = u_pix_bnd_drive + u_pix_offset_orth[i_orth]

        # Early rejection if fully off detector
        if P_bnd[-1] <= u_det_bnd[0] or P_bnd[0] >= u_det_bnd[-1]:
            continue

        i_pix = i_det = 0
        while i_pix < n_pix_drive and i_det < n_dets:
            left  = max(P_bnd[i_pix],     u_det_bnd[i_det])
            right = min(P_bnd[i_pix + 1], u_det_bnd[i_det + 1])

            if right > left:
                img_rot[i_pix, i_orth] += proj[i_det] * (right - left) / inv
            
            # Advance to next pixel or detector
            if P_bnd[i_pix + 1] < u_det_bnd[i_det + 1]:
                i_pix += 1
            else:
                i_det += 1
                

def dd_bp_par_2d(sino, angs, n_x, n_y, d_pix=1.0, d_det=1.0):
    """
    2D parallel-beam distance-driven back-projection.

    Reconstructs a 2D image from a parallel-beam sinogram using
    distance-driven back-projection. The algorithm selects the
    driving axis (X or Y) depending on the projection angle for
    better sampling along the dominant direction.

    Parameters
    ----------
    sino : ndarray, shape (n_angles, n_dets)
        Parallel-beam sinogram.
    angs : ndarray, shape (n_angles,)
        Projection angles in radians.
    n_x : int
        Number of pixels along X axis.
    n_y : int
        Number of pixels along Y axis.
    d_pix : float, optional
        Pixel width. Default is 1.0.
    d_det : float, optional
        Detector bin width. Default is 1.0.

    Returns
    -------
    img : ndarray, shape (n_x, n_y)
        Reconstructed 2D image.

    Notes
    -----
    - Selects X-driven or Y-driven back-projection depending on angle to reduce sampling artifacts.
    - Uses pixel boundary overlaps for accurate distance-driven weighting.
    """
    
    sino = np.asarray(sino, dtype=np.float32)
    n_angs, n_dets = sino.shape
    img = np.zeros((n_x, n_y), dtype=np.float32)

    # Pixel boundaries
    x_bnd = d_pix * (np.arange(n_x + 1) - n_x / 2.0)
    y_bnd = d_pix * (np.arange(n_y + 1) - n_y / 2.0)

    # Pixel centers
    x_cnt = (x_bnd[:-1] + x_bnd[1:]) / 2
    y_cnt = (y_bnd[:-1] + y_bnd[1:]) / 2

    # Detector boundaries
    u_det_bnd = d_det * (np.arange(n_dets + 1) - n_dets / 2.0)

    for i_ang, theta in enumerate(angs):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        proj = sino[i_ang]

        # X-driven
        if abs(cos_t) >= abs(sin_t):
            n_pix_drive, n_pix_orth = n_x, n_y
            inv = max(abs(cos_t), 1e-12)
            
            u_pix_bnd_drive = cos_t * x_bnd
            u_pix_offset_orth = -sin_t * y_cnt

            img_rot = img
            
        # Y-driven
        else:
            n_pix_drive, n_pix_orth = n_y, n_x
            inv = max(abs(sin_t), 1e-12)
            
            u_pix_bnd_drive = -sin_t * y_bnd
            u_pix_offset_orth = cos_t * x_cnt

            img_rot = img.T 
            
            
        if u_pix_bnd_drive[1] < u_pix_bnd_drive[0]:
            u_pix_bnd_drive = u_pix_bnd_drive[::-1]
            img_rot = img_rot[::-1, :]
            
        _dd_bp_par_2d_sweep(img_rot,proj,u_pix_bnd_drive,u_pix_offset_orth,u_det_bnd,n_pix_drive,n_pix_orth,inv)
            
    return img/n_angs/d_pix


def _dd_bp_fan_2d_sweep(img, proj, u_pix_bnd_drive, u_pix_offset_orth,
                        det_bnd, n_pix_drive, n_pix_orth, backproj_scale):
    """
    Distance-driven sweep for fan-beam back-projection along one angle.

    Accumulates fan-beam projection contributions into image pixels along the
    driving axis. Each pixel contribution is weighted by the detector overlap
    and corrected for fan-beam magnification.

    Parameters
    ----------
    img : ndarray, shape (n_pix_drive, n_pix_orth)
        Subarray of the image for the current driving axis (may be a view).
    proj : ndarray, shape (n_dets,)
        Fan-beam projection for the current angle.
    u_pix_bnd_drive : ndarray, shape (n_pix_drive + 1,)
        Pixel boundaries along the driving axis, projected onto the detector.
    u_pix_offset_orth : ndarray, shape (n_pix_orth,)
        Orthogonal pixel offsets for each row/column.
    det_bnd : ndarray, shape (n_dets + 1,)
        Detector bin boundaries.
    n_pix_drive : int
        Number of pixels along the driving axis.
    n_pix_orth : int
        Number of pixels along the orthogonal axis.
    backproj_scale : ndarray, shape (n_pix_orth,)
        Fan-beam magnification factor for each orthogonal pixel
        (typically DSD / (DSO - u)).

    Notes
    -----
    - Implements distance-driven back-projection for fan-beam CT.
    - Early rejection is applied for pixels fully outside the detector.
    - Magnification ensures approximate intensity conservation in fan-beam geometry.
    """
    
    n_dets = det_bnd.size - 1
    
    # Loop over orthogonal pixels
    for i_orth in range(n_pix_orth):
        u_pix_bnd = u_pix_bnd_drive + u_pix_offset_orth[i_orth]
        scale = backproj_scale[i_orth]

        # Early rejection if fully off detector
        if u_pix_bnd[-1] <= det_bnd[0] or u_pix_bnd[0] >= det_bnd[-1]:
            continue

        i_pix = i_det = 0
        while i_pix < n_pix_drive and i_det < n_dets:
            left  = max(u_pix_bnd[i_pix],     det_bnd[i_det])
            right = min(u_pix_bnd[i_pix+1],   det_bnd[i_det+1])

            if right > left:
                # Accumulate backprojection, correcting for fan-beam magnification
                img[i_pix, i_orth] += proj[i_det] * (right - left) / scale

            # Advance to next pixel or detector
            if u_pix_bnd[i_pix+1] < det_bnd[i_det+1]:
                i_pix += 1
            else:
                i_det += 1


def dd_bp_fan_2d(sino, Angs, n_x, n_y, DSO, DSD, d_pix=1.0, d_det=1.0):
    """
    2D fan-beam distance-driven back-projection.

    Reconstructs a 2D image from a fan-beam sinogram using
    distance-driven back-projection. Supports X-driven or Y-driven
    selection depending on projection angle. Corrects for
    fan-beam magnification to approximate intensity conservation.

    Parameters
    ----------
    sino : ndarray, shape (n_angles, n_dets)
        Fan-beam sinogram.
    Angs : ndarray, shape (n_angles,)
        Projection angles in radians.
    n_x : int
        Number of pixels along X axis.
    n_y : int
        Number of pixels along Y axis.
    DSO : float
        Source-to-origin distance.
    DSD : float
        Source-to-detector distance.
    d_pix : float, optional
        Pixel width. Default is 1.0.
    d_det : float, optional
        Detector bin width. Default is 1.0.

    Returns
    -------
    img : ndarray, shape (n_x, n_y)
        Reconstructed 2D image.

    Notes
    -----
    - Uses distance-driven pixel-detector overlap weighting for fan-beam back-projection.
    - Supports X-driven / Y-driven sweeping depending on projection angle.
    - Magnification correction ensures approximate intensity conservation for finite DSO/DSD.
    """
    
    sino = np.asarray(sino, dtype=np.float32)
    n_angs, n_dets = sino.shape
    img = np.zeros((n_x, n_y), dtype=np.float32)

    # Pixel boundaries and centers
    X_bnd = d_pix * (np.arange(n_x+1) - n_x/2)
    Y_bnd = d_pix * (np.arange(n_y+1) - n_y/2)
    X_cnt = 0.5 * (X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5 * (Y_bnd[:-1] + Y_bnd[1:])

    # Detector boundaries
    det_bnd = d_det * (np.arange(n_dets + 1) - n_dets/2)

    for i_ang, theta in enumerate(Angs):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        proj = sino[i_ang]

        # Decide driving axis
        if abs(cos_t) >= abs(sin_t):
            n_pix_drive, n_pix_orth = n_x, n_y
            u_pix_bnd_drive = cos_t * X_bnd
            u_pix_offset_orth = -sin_t * Y_cnt
            img_view = img
        else:
            n_pix_drive, n_pix_orth = n_y, n_x
            u_pix_bnd_drive = -sin_t * Y_bnd
            u_pix_offset_orth = cos_t * X_cnt
            img_view = img.T

        # Reverse if projected pixels are decreasing
        if u_pix_bnd_drive[-1] < u_pix_bnd_drive[0]:
            u_pix_bnd_drive = u_pix_bnd_drive[::-1]
            img_view = img_view[::-1, :]

        # Magnification factor (inverse of forward-projection scaling)
        backproj_scale = DSD / (DSO - u_pix_offset_orth)

        _dd_bp_fan_2d_sweep(img_view, proj, u_pix_bnd_drive, u_pix_offset_orth,
                            det_bnd, n_pix_drive, n_pix_orth, backproj_scale)

    return img/n_angs/d_pix



def _dd_fp_cone_3d_sweep(
    sino, vol,
    u_pix_bnd_drive, u_pix_offset_y, v_pix_offset_z,
    det_u_bnd, det_v_bnd,
    n_pix_drive, n_pix_y, n_pix_z,
    proj_scale_y, proj_scale_z,
    i_ang
):
    """
    Distance-driven cone-beam sweep for one projection angle.

    Accumulates voxel contributions into (u, v) detector bins.
    """

    n_det_u = det_u_bnd.size - 1
    n_det_v = det_v_bnd.size - 1

    for iy in range(n_pix_y):
        for iz in range(n_pix_z):

            voxel_line = vol[:, iy, iz]

            # Apply orthogonal offsets
            u_pix_bnd = u_pix_bnd_drive + u_pix_offset_y[iy]
            v_pix = v_pix_offset_z[iz]

            # Magnification correction (cone-beam)
            proj_scale = proj_scale_y[iy] * proj_scale_z[iz]

            i_pix = i_det_u = 0
            while i_pix < n_pix_drive and i_det_u < n_det_u:
                left_u  = max(u_pix_bnd[i_pix],   det_u_bnd[i_det_u])
                right_u = min(u_pix_bnd[i_pix+1], det_u_bnd[i_det_u+1])

                if right_u > left_u:
                    # v overlap (point-like along drive axis)
                    for i_det_v in range(n_det_v):
                        left_v  = max(v_pix, det_v_bnd[i_det_v])
                        right_v = min(v_pix, det_v_bnd[i_det_v+1])

                        if right_v > left_v:
                            sino[i_ang, i_det_u, i_det_v] += (
                                voxel_line[i_pix]
                                * (right_u - left_u)
                                * (right_v - left_v)
                                / proj_scale
                            )

                if u_pix_bnd[i_pix+1] < det_u_bnd[i_det_u+1]:
                    i_pix += 1
                else:
                    i_det_u += 1


def dd_fp_cone_3d(
    vol, Angs,
    n_det_u, n_det_v,
    DSO, DSD,
    d_pix=1.0, d_det_u=1.0, d_det_v=1.0
):
    """
    Distance-driven cone-beam forward projection (flat-panel).

    Parameters
    ----------
    vol : ndarray, shape (nX, nY, nZ)
        3D image volume.
    Angs : ndarray
        Projection angles (radians).
    n_det_u, n_det_v : int
        Detector dimensions.
    DSO, DSD : float
        Source-to-origin and source-to-detector distances.
    """

    nX, nY, nZ = vol.shape
    sino = np.zeros((len(Angs), n_det_u, n_det_v), dtype=np.float32)

    # Image voxel boundaries
    X_bnd = d_pix * (np.arange(nX + 1) - nX / 2)
    Y_bnd = d_pix * (np.arange(nY + 1) - nY / 2)
    Z_bnd = d_pix * (np.arange(nZ + 1) - nZ / 2)

    X_cnt = 0.5 * (X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5 * (Y_bnd[:-1] + Y_bnd[1:])
    Z_cnt = 0.5 * (Z_bnd[:-1] + Z_bnd[1:])

    # Detector boundaries
    det_u_bnd = d_det_u * (np.arange(n_det_u + 1) - n_det_u / 2)
    det_v_bnd = d_det_v * (np.arange(n_det_v + 1) - n_det_v / 2)

    for i_ang, theta in enumerate(Angs):
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Driving axis decision (same as 2D)
        if abs(cos_t) >= abs(sin_t):
            # X-driven
            n_pix_drive = nX
            n_pix_y = nY
            n_pix_z = nZ

            u_pix_bnd_drive = cos_t * X_bnd
            u_pix_offset_y = -sin_t * Y_cnt
            v_pix_offset_z = Z_cnt

            volP = vol
        else:
            # Y-driven
            n_pix_drive = nY
            n_pix_y = nX
            n_pix_z = nZ

            u_pix_bnd_drive = -sin_t * Y_bnd
            u_pix_offset_y = cos_t * X_cnt
            v_pix_offset_z = Z_cnt

            volP = vol.transpose(1, 0, 2)

        # Enforce monotonicity
        if u_pix_bnd_drive[-1] < u_pix_bnd_drive[0]:
            u_pix_bnd_drive = u_pix_bnd_drive[::-1]
            volP = volP[::-1, :, :]

        # Cone-beam magnification
        proj_scale_y = DSD / (DSO - u_pix_offset_y)
        proj_scale_z = DSD / DSO  # z magnification (flat panel assumption)

        _dd_fp_cone_3d_sweep(
            sino, volP,
            u_pix_bnd_drive,
            u_pix_offset_y,
            v_pix_offset_z,
            det_u_bnd, det_v_bnd,
            n_pix_drive, n_pix_y, n_pix_z,
            proj_scale_y, proj_scale_z,
            i_ang
        )

    return sino

