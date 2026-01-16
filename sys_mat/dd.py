#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 06:34:23 2026

@author: pvargas21
"""

import numpy as np

def _dd_fp_sweep(sino,img_trm,u_pixs_drive_bnd,u_pixs_orthg_off,
                          u_det_bnd,n_pix_drive,n_pix_orthg,rays_scl,i_ang):
    """
    Distance-driven sweep kernel for 2D parallel-beam forward projection.

    This function performs the core distance-driven accumulation for a single
    projection angle. It assumes that pixel boundaries projected onto the
    detector coordinate are monotonic along the driving axis, enabling a
    linear-time sweep over pixels and detector bins.

    Geometry:
        - Parallel-beam
        - Detector coordinate u is linear and orthogonal to ray direction

    Parameters
    ----------
    sino : ndarray, shape (n_angles, n_det)
        Output sinogram array to be accumulated in-place.
    img_trm : ndarray, shape (n_pix_drive, n_pix_orth)
        Image data after transpose and/or flip so that axis 0 corresponds
        to the driving (sweep) axis.
    u_pixs_drive_bnd : ndarray, shape (n_pix_drive + 1,)
        Projected pixel boundary coordinates along the driving axis in
        detector space, independent of orthogonal pixel index.
    u_pixs_orthg_off : ndarray, shape (n_pix_orth,)
        Orthogonal pixel-center offsets projected onto the detector coordinate.
    u_det_bnd : ndarray, shape (n_det + 1,)
        Detector bin boundary coordinates in detector space.
    n_pix_drive : int
        Number of pixels along the driving axis.
    n_pix_orthg : int
        Number of pixels along the orthogonal axis.
    rays_scl : ndarray, shape (n_det)
        Projection scaling factor (typically |cos(theta)| or |sin(theta)|)
        accounting for pixel width along the ray direction.
    i_ang : int
        Index of the current projection angle in the sinogram.

    Notes
    -----
    - The sweep assumes strictly monotonic projected pixel boundaries.
    - Contributions outside the detector range are implicitly discarded.
    - This function does not allocate memory and updates `sino` in-place.
    """

    n_det = u_det_bnd.size - 1


    # Loop over orthogonal pixel lines
    for i_orth in range(n_pix_orthg):

        #Hoisting pointers for explicit LICM
        u_pix_orthg_off = u_pixs_orthg_off[i_orth]

        img_vec = img_trm[:, i_orth]
        ray_scl = rays_scl[i_orth]

        i_pix = 0
        i_det = 0
        while i_pix < n_pix_drive and i_det < n_det:
            
            # Left edge of overlap interval
            # Actual projected pixel boundaries for this pixel row/column
            pix_bnd_l = u_pixs_drive_bnd[i_pix] + u_pix_orthg_off
            det_bnd_l = u_det_bnd[i_det]
            overlap_l  = pix_bnd_l if pix_bnd_l > det_bnd_l else det_bnd_l

            # Right edge of overlap interval
            # Actual projected pixel boundaries for this pixel row/column
            pix_bnd_r = u_pixs_drive_bnd[i_pix + 1] + u_pix_orthg_off
            det_bnd_r = u_det_bnd[i_det + 1]
            overlap_r  = pix_bnd_r if pix_bnd_r < det_bnd_r else det_bnd_r
        
            # Accumulate overlap contribution
            if overlap_r > overlap_l:
                sino[i_ang, i_det] += (img_vec[i_pix]* (overlap_r - overlap_l)*ray_scl)

            # Advance whichever interval ends first
            if pix_bnd_r < det_bnd_r:
                i_pix += 1
            else:
                i_det += 1

def compute_projection_geometry(img, nx_pix, ny_pix,
                                x_pixs_bnd, y_pixs_bnd,
                                x_pixs_cnt, y_pixs_cnt,
                                c_ang, s_ang, is_fan=False,
                                DSO=1.0, DSD=1.0):
    """
    Compute the driving geometry for a single projection angle, modular
    for both parallel-beam and fan-beam distance-driven forward projection.

    Returns:
        n_pix_drive        : int, number of pixels along driving axis
        n_pix_orthg        : int, number of pixels along orthogonal axis
        img_trm            : ndarray view (transposed if Y-driven)
        u_pixs_drive_bnd   : ndarray, projected pixel boundaries along driving axis
        u_pixs_orthg_off   : ndarray, orthogonal pixel offsets
        ray_scale          : float or ndarray, scaling factor for sweep
    """
    
    # Determine driving axis
    if abs(s_ang) >= abs(c_ang):
        # X-driven
        n_pix_drive, n_pix_orthg = nx_pix, ny_pix
        
        # Scales projected pixel overlap along X for oblique rays
        ray_scale = abs(s_ang)

        # Project pixel boundaries along driving axis onto detector
        # Each X pixel boundary is projected along the ray direction
        u_pixs_drive_bnd = -s_ang * x_pixs_bnd

        # Orthogonal pixel-center offset added to base detector projection
        u_pixs_orthg_off = c_ang * y_pixs_cnt

        #No transformation need  
        img_trm = img  

    else:
        # Y-driven
        n_pix_drive, n_pix_orthg = ny_pix, nx_pix
        
        # Scales projected pixel overlap along Y for oblique rays
        ray_scale = abs(c_ang)

        # Project pixel boundaries along driving axis onto detector
        u_pixs_drive_bnd = c_ang * y_pixs_bnd

        # Orthogonal pixel-center offset added to base detector projection
        u_pixs_orthg_off = -s_ang * x_pixs_cnt

        # Transposes image so axis 0 correspondsto the driving (sweep) axis
        img_trm = img.T  

    # Ensure monotonic increasing for sweep
    if u_pixs_drive_bnd[1] < u_pixs_drive_bnd[0]:
        u_pixs_drive_bnd = u_pixs_drive_bnd[::-1]
        img_trm = img_trm[::-1, :]

    # Fan-beam magnification correction.
    # Values are inverted to allow multiplicaion in hot loop
    if is_fan:
        # Fan-beam magnification correction:
        # Each pixel's overlap is divided by proj_scale
        # to approximate true line integral along the ray.
        # Multiply by ray_scale to convert driving-axis projection
        # to ray length        
        magnification = DSD / (DSO - u_pixs_orthg_off)
        rays_scale = 1/(magnification * ray_scale)
    else:
        #Mirroring fanbeam structure
        rays_scale = np.full(n_pix_orthg, 1.0/ray_scale, dtype=np.float32)


    return n_pix_drive, n_pix_orthg, img_trm, u_pixs_drive_bnd, u_pixs_orthg_off, rays_scale


def dd_fp_par_2d(img, angles, n_dets, d_det=1.0, d_pix=1.0):
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
    angles : ndarray, shape (n_angles,)
        Projection angles in radians.
    n_det : int
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

    nx_pix, ny_pix = img.shape
    sino = np.zeros((angles.size, n_dets), dtype=np.float32)

    # Define pixel boundaries and centers in image space
    # Centered at origin (0,0)
    # x_pixs_bnd, y_pixs_bnd: positions of pixel edges along X and Y
    x_pixs_bnd = d_pix*(np.arange(nx_pix+1) - nx_pix/2)
    y_pixs_bnd = d_pix*(np.arange(ny_pix+1) - ny_pix/2)
 
    # x_pixs_cnt, y_pixs_cnt: positions of pixel centers
    x_pixs_cnt = (x_pixs_bnd[:-1] + x_pixs_bnd[1:])/2
    y_pixs_cnt = (y_pixs_bnd[:-1] + y_pixs_bnd[1:])/2

    # Detector bin boundaries along the fan-beam arc
    # Centered at u = 0
    u_det_bnd = d_det * (np.arange(n_dets + 1) - n_dets / 2.0)

    # Precompute trig functions for all angles
    c_angs = np.cos(angles)
    s_angs = np.sin(angles)

    #Loop through projection angles
    for i_ang, (c_ang,s_ang) in enumerate(zip(c_angs,s_angs)):
       
        n_pix_drive, n_pix_orthg, img_trm, u_pixs_drive_bnd, u_pixs_orthg_off, rays_scale = \
            compute_projection_geometry(img, nx_pix, ny_pix,
                                x_pixs_bnd, y_pixs_bnd,
                                x_pixs_cnt, y_pixs_cnt,
                                c_ang, s_ang,
                                is_fan=False)
    
        _dd_fp_sweep(sino, img_trm, u_pixs_drive_bnd, u_pixs_orthg_off,
                         u_det_bnd, n_pix_drive, n_pix_orthg, rays_scale, i_ang)

    #Return sino normalized with pixel and detector size
    return sino*d_pix/d_det


def dd_fp_fan_2d(img, Angs, n_dets, DSO, DSD, d_det=1.0, d_pix=1.0):
    """
    Distance-driven fan-beam forward projection for 2D CT.

    Computes the sinogram of a 2D image for a set of projection angles
    in fan-beam geometry.

    Parameters
    ----------
    img : ndarray, shape (nX, nY)
        2D image to project.
    Angs : ndarray, shape (n_angles,)
        Projection angles in radians.
    nDets : int
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
    
    nx_pix, ny_pix = img.shape
    sino = np.zeros((len(Angs), n_dets), dtype=np.float32)

    # Define pixel boundaries and centers in image space
    # Centered at origin (0,0)
    # x_pixs_bnd, y_pixs_bnd: positions of pixel edges along X and Y
    x_pixs_bnd = d_pix*(np.arange(nx_pix+1) - nx_pix/2)
    y_pixs_bnd = d_pix*(np.arange(ny_pix+1) - ny_pix/2)
 
    # x_pixs_cnt, y_pixs_cnt: positions of pixel centers
    x_pixs_cnt = (x_pixs_bnd[:-1] + x_pixs_bnd[1:])/2
    y_pixs_cnt = (y_pixs_bnd[:-1] + y_pixs_bnd[1:])/2

    # Detector bin boundaries along the fan-beam arc
    # Centered at u = 0
    u_det_bnd = d_det * (np.arange(n_dets + 1) - n_dets / 2.0)

    # Precompute trig functions for all angles
    c_angs = np.cos(Angs)
    s_angs = np.sin(Angs)

    #Loop through projection angles
    for i_ang, (c_ang,s_ang) in enumerate(zip(c_angs,s_angs)):

        n_pix_drive, n_pix_orthg, img_trm, u_pixs_drive_bnd, u_pixs_orthg_off, rays_scale = \
           compute_projection_geometry(img, nx_pix, ny_pix,
                               x_pixs_bnd, y_pixs_bnd,
                               x_pixs_cnt, y_pixs_cnt,
                               c_ang, s_ang,
                                is_fan=True, DSO=DSO, DSD=DSD)

        # Sweep along driving axis
        _dd_fp_sweep(sino, img_trm, u_pixs_drive_bnd, u_pixs_orthg_off,
                            u_det_bnd, n_pix_drive, n_pix_orthg, rays_scale, i_ang)

    #Return sino normalized with pixel and detector size
    return sino*d_pix/d_det




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

