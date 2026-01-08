# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 08:57:56 2026

@author: varga
"""


import numpy as np


def _dd_fp_par_sweep(sino,img_rot,u_pix_bnd_drive,u_pix_offset_orth,
                          u_det_bnd,n_pix_drive,n_pix_orth,proj_scale,i_ang):
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
    img_rot : ndarray, shape (n_pix_drive, n_pix_orth)
        Image data after transpose and/or flip so that axis 0 corresponds
        to the driving (sweep) axis.
    u_pix_bnd_drive : ndarray, shape (n_pix_drive + 1,)
        Projected pixel boundary coordinates along the driving axis in
        detector space, independent of orthogonal pixel index.
    u_pix_offset_orth : ndarray, shape (n_pix_orth,)
        Orthogonal pixel-center offsets projected onto the detector coordinate.
    u_det_bnd : ndarray, shape (n_det + 1,)
        Detector bin boundary coordinates in detector space.
    n_pix_drive : int
        Number of pixels along the driving axis.
    n_pix_orth : int
        Number of pixels along the orthogonal axis.
    proj_scale : float
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
    for i_orth in range(n_pix_orth):
        pix_line = img_rot[:, i_orth]

        # Actual projected pixel boundaries for this pixel row/column
        u_pix_bnd = u_pix_bnd_drive + u_pix_offset_orth[i_orth]

        i_pix = i_det = 0
        while i_pix < n_pix_drive and i_det < n_det:

            # Overlap between projected pixel and detector bin
            left  = max(u_pix_bnd[i_pix],   u_det_bnd[i_det])
            right = min(u_pix_bnd[i_pix+1], u_det_bnd[i_det+1])

            if right > left:
                sino[i_ang, i_det] += pix_line[i_pix] * (right - left) / proj_scale

            # Advance the boundary that ends first
            if u_pix_bnd[i_pix+1] < u_det_bnd[i_det+1]:
                i_pix += 1
            else:
                i_det += 1


def dd_fp_par_2d(img, angles, n_det, d_pix=1.0, d_det=1.0):
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
    d_pix : float, optional
        Pixel width (default is 1.0).
    d_det : float, optional
        Detector bin width (default is 1.0).

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

    n_x, n_y = img.shape
    sino = np.zeros((angles.size, n_det), dtype=np.float32)

    # Pixel boundary coordinates
    x_bnd = d_pix * (np.arange(n_x + 1) - n_x / 2.0)
    y_bnd = d_pix * (np.arange(n_y + 1) - n_y / 2.0)

    # Pixel center coordinates
    x_cnt = 0.5 * (x_bnd[:-1] + x_bnd[1:])
    y_cnt = 0.5 * (y_bnd[:-1] + y_bnd[1:])

    # Detector bin boundaries (detector coordinate u)
    u_det_bnd = d_det * (np.arange(n_det + 1) - n_det / 2.0)

    for i_ang, theta in enumerate(angles):
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Choose driving axis so projected boundaries are well-conditioned
        if abs(cos_t) >= abs(sin_t):
            # X-driven
            n_pix_drive, n_pix_orth = n_x, n_y
            proj_scale = max(abs(cos_t), 1e-12)

            # Project pixel boundaries onto detector axis
            u_pix_bnd_drive = cos_t * x_bnd
            
            # Orthogonal pixel-center offset added to base detector projection
            u_pix_offset_orth = -sin_t * y_cnt
            img_rot = img #No rotation needed


        else:
            # Y-driven
            n_pix_drive, n_pix_orth = n_y, n_x
            proj_scale = max(abs(sin_t), 1e-12)

            # Project pixel boundaries onto detector axis
            u_pix_bnd_drive = -sin_t * y_bnd

            # Orthogonal pixel-center offset added to base detector projection
            u_pix_offset_orth = cos_t * x_cnt

            # Transposes image so axis 0 correspondsto the driving (sweep) axis
            img_rot = img.T 

        # Ensure monotonicity
        if u_pix_bnd_drive[1] < u_pix_bnd_drive[0]:
            u_pix_bnd_drive = u_pix_bnd_drive[::-1]
            img_rot = img_rot[::-1, :]


        _dd_fp_par_sweep(
            sino,
            img_rot,
            u_pix_bnd_drive,
            u_pix_offset_orth,
            u_det_bnd,
            n_pix_drive,
            n_pix_orth,
            proj_scale,
            i_ang
        )

    return sino


def _dd_fp_fan_2d_sweep(sino, img, u_pix_bnd_drive, u_pix_offset_orth, 
                        det_bnd, n_pix_drive, n_pix_orth, proj_scale_arr, i_ang):
    """
    Distance-driven fan-beam sweep for one projection angle.

    Accumulates line integrals from image pixels into sinogram bins along the
    driving axis. Corrects for fan-beam magnification to approximately conserve intensity.

    Parameters
    ----------
    sino : ndarray, shape (n_angles, n_dets)
        Sinogram array to accumulate into.
    img : ndarray, shape (n_pix_drive, n_pix_orth)
        Image array for current projection (rows = driving axis).
    u_pix_bnd_drive : ndarray, shape (n_pix_drive + 1,)
        Pixel boundaries along the driving axis, projected to the detector plane.
    u_pix_offset_orth : ndarray, shape (n_pix_orth,)
        Orthogonal pixel offsets along the detector (e.g., Y contribution for X-driven).
    det_bnd : ndarray, shape (n_dets + 1,)
        Detector bin boundaries.
    n_pix_drive : int
        Number of pixels along the driving axis.
    n_pix_orth : int
        Number of pixels along the orthogonal axis.
    i_ang : int
        Index of the current projection angle.

    Notes
    -----
    - Corrects each pixel's contribution using the magnification factor:
      magnification = DSD / (DSO - u_pix_offset_orth[i_orth])
    - For coarse pixels or oblique angles, the distance-driven approximation
      underestimates line integrals along diagonal rays. Subpixel splitting
      is recommended for improved accuracy.
    """
    
    n_dets = det_bnd.size - 1
    
    
    # Loop over orthogonal pixels (rows or columns depending on driving axis)
    for i_orth in range(n_pix_orth):
        pix_line = img[:, i_orth]  # pixel values along driving axis
        proj_scale = proj_scale_arr[i_orth]
        
        # Add orthogonal offset to driving axis boundaries
        u_pix_bnd = u_pix_bnd_drive + u_pix_offset_orth[i_orth]

        # Distance-driven sweep along driving axis
        i_pix = i_det = 0
        while i_pix < n_pix_drive and i_det < n_dets:
            # Overlap between projected pixel and detector bin
            left  = max(u_pix_bnd[i_pix],     det_bnd[i_det])
            right = min(u_pix_bnd[i_pix+1],   det_bnd[i_det+1])
            
            if right > left:
                # Magnification of this pixel due to fan geometry
                # pixel length in detector space = Δu * (DSO - y) / DSD
                
                # Accumulate line integral, correcting for fan-beam magnification
                sino[i_ang, i_det] += pix_line[i_pix] * (right - left) / proj_scale

            # Advance to next pixel or detector bin
            if u_pix_bnd[i_pix+1] < det_bnd[i_det+1]:
                i_pix += 1
            else:
                i_det += 1

#def dd_fp_fan_2d(img, angles, n_det, DSO, DSD, d_pix=1.0, d_det=1.0):
def dd_fp_fan_2d(img, Angs, n_dets, DSO, DSD, d_pix=1.0, d_det=1.0):
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
    dPix : float, optional
        Pixel width in image units. Default is 1.0.
    dDet : float, optional
        Detector bin width. Default is 1.0.

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
    
    nX, nY = img.shape
    sino = np.zeros((len(Angs), n_dets), dtype=np.float32)

    # Pixel boundaries and centers
    X_bnd = d_pix * (np.arange(nX+1) - nX/2)
    Y_bnd = d_pix * (np.arange(nY+1) - nY/2)
    X_cnt = 0.5 * (X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5 * (Y_bnd[:-1] + Y_bnd[1:])

    # Detector bin boundaries
    u_det_bnd = d_det * (np.arange(n_dets + 1) - n_dets / 2.0)

    for i_ang, theta in enumerate(Angs):
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Decide driving axis (closest to horizontal/vertical)
        if abs(cos_t) >= abs(sin_t):
            # X-driven
            n_pix_drive, n_pix_orth = nX, nY
            
            u_pix_bnd_drive = cos_t * X_bnd      # base projected boundaries
            u_pix_offset_orth = -sin_t * Y_cnt   # orthogonal offsets
            
            imgP = img
        else:
            # Y-driven
            n_pix_drive, n_pix_orth = nY, nX
            
            u_pix_bnd_drive = -sin_t * Y_bnd
            u_pix_offset_orth = cos_t * X_cnt
            
            imgP = img.T

        # If projected pixels are decreasing along detector, reverse them
        if u_pix_bnd_drive[-1] < u_pix_bnd_drive[0]:
            u_pix_bnd_drive = u_pix_bnd_drive[::-1]
            imgP = imgP[::-1, :]

        proj_scale = DSD / (DSO - u_pix_offset_orth)
        # Sweep along driving axis
        _dd_fp_fan_2d_sweep(sino, imgP, u_pix_bnd_drive, u_pix_offset_orth,
                            u_det_bnd, n_pix_drive, n_pix_orth, proj_scale, i_ang)

    return sino




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
            inv = max(d_det*abs(cos_t), 1e-12)
            
            u_pix_bnd_drive = cos_t * x_bnd
            u_pix_offset_orth = -sin_t * y_cnt

            img_rot = img
            
        # Y-driven
        else:
            n_pix_drive, n_pix_orth = n_y, n_x
            inv = max(d_det*abs(sin_t), 1e-12)
            
            u_pix_bnd_drive = -sin_t * y_bnd
            u_pix_offset_orth = cos_t * x_cnt

            img_rot = img.T 
            
            
        if u_pix_bnd_drive[1] < u_pix_bnd_drive[0]:
            u_pix_bnd_drive = u_pix_bnd_drive[::-1]
            img_rot = img_rot[::-1, :]
            
        _dd_bp_par_2d_sweep(img_rot,proj,u_pix_bnd_drive,u_pix_offset_orth,u_det_bnd,n_pix_drive,n_pix_orth,inv)
            
    return img


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

    return img





















def siddons_fp_2d(img, Angs, nDets, dPix=1.0, dDet=1.0):
    nX, nY = img.shape
    sino = np.zeros((Angs.size, nDets), dtype=np.float32)

    # Image edges (centered)
    X_bnd = dPix * (np.arange(nX+1) - nX/2.0)
    Y_bnd = dPix * (np.arange(nY+1) - nY/2.0)

    # Detector bins (centered)
    Dets_cnt = dDet * (np.arange(nDets) - nDets/2.0 + 0.5)

    for iAng, ang in enumerate(Angs):
        dx, dy = np.cos(ang), np.sin(ang)

        for iDet, det_cnt in enumerate(Dets_cnt):
            # Setup ray origin *ON detector array, perpendicular to ray direction*
            # For parallel beam: select t so that for t = 0, detector axis at image center
            # ray passes through (x0, y0) = (det_center_offset along normal)
            x0 = det_cnt * -dy
            y0 = det_cnt * dx

            # Calculate entry and exit parameters
            txs, tys = np.array([-np.inf, np.inf]), np.array([-np.inf, np.inf])
            if dx != 0:
                txs = (X_bnd - x0) / dx
            if dy != 0:
                tys = (Y_bnd - y0) / dy

            t0 = max(min(txs[0], txs[-1]), min(tys[0], tys[-1]))
            t1 = min(max(txs[0], txs[-1]), max(tys[0], tys[-1]))

            if t1 <= t0 or np.isinf(t0) or np.isinf(t1):
                continue
            # Find all boundary crossings
            cross_list = []
            if dx != 0:
                cross_list += list((X_bnd - x0) / dx)
            if dy != 0:
                cross_list += list((Y_bnd - y0) / dy)
            crosses = np.array(
                sorted([t for t in cross_list if t0 <= t <= t1]))
            if crosses.size < 2:
                continue
            for seg in range(crosses.size-1):
                tc0, tc1 = crosses[seg], crosses[seg+1]
                if tc1 <= tc0:
                    continue
                xc = x0 + 0.5*(tc0+tc1)*dx
                yc = y0 + 0.5*(tc0+tc1)*dy
                ix = int(np.floor((xc - X_bnd[0]) / dPix))
                iy = int(np.floor((yc - Y_bnd[0]) / dPix))
                if 0 <= ix < nX and 0 <= iy < nY:
                    sino[iAng, iDet] += img[ix, iy] * abs(tc1-tc0)
    return sino


def joseph_fp_2d(img, Angs, nDets, dPix=1.0, dDet=1.0):
    """
    Joseph's ray-interpolation forward projector for 2D parallel-beam CT.

    Parameters:
        img     : ndarray, shape (nX, nY)
        angles  : ndarray of projection angles [radians]
        nDets   : number of detector bins
        dPix    : pixel size (width)
        dDet    : detector bin width

    Returns:
        sino    : ndarray (len(angles), nDets)
    """
    img = np.ascontiguousarray(img, dtype=np.float32)
    Angs = np.asarray(Angs)
    nX, nY = img.shape
    sino = np.zeros((Angs.size, nDets), dtype=np.float32)

    # Grid: centered at zero, units in physical space
    x0 = -dPix*(nX + 1)/2.0  # First pixel center (x)
    y0 = -dPix*(nY + 1)/2.0  # First pixel center (y)

    Dets_cnt = dDet*(np.arange(nDets) - nDets / 2.0 + 0.5)

    # Project image grid boundaries onto the ray
    # Ray length: covers diagonal of image for safety
    L = dPix * max(nX, nY) * 2

    # Find t range so that we cover the whole image
    t0 = -L / 2
    t1 = L / 2

    for iAng, angle in enumerate(Angs):
        ang_cos, ang_sin = np.cos(angle), np.sin(angle)

        # Ray direction is [cos_a, sin_a], detector axis is [-sin_a, cos_a]
        for iDet, det_cnt in enumerate(Dets_cnt):
            # Ray passes through (x_s, y_s)

            x_s = -ang_sin * det_cnt
            y_s = ang_cos * det_cnt

            # Step size along ray (Joseph typically steps in 1-pixel increments)
            step = dPix / max(abs(ang_cos), abs(ang_sin))

            t = t0
            while t <= t1:
                # Current position along ray
                x = x_s + ang_cos * t
                y = y_s + ang_sin * t

                # Convert to pixel index
                ix = (x - x0) / dPix
                iy = (y - y0) / dPix

                if 0 <= ix < nX-1 and 0 <= iy < nY-1:
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
                    sino[iAng, iDet] += val * step
                t += step
    return sino


def separable_footprint_fp_2d(img, Angs, nDets, dPix=1.0, dDet=1.0):
    """
    Separable footprints forward projector for 2D parallel-beam CT.

    Implements the separable footprint model where each pixel is projected
    as an axis-aligned rectangle (“footprint”) onto the detector array.

    Parameters:
        img     : ndarray shape (nX, nY)
        Angs  : ndarray of projection angles (radians)
        nDets   : number of detector bins
        dPix    : pixel width
        dDet    : detector bin width

    Returns:
        sino    : ndarray shape (len(angles), nDets)
    """
    nX, nY = img.shape
    sino = np.zeros((Angs.size, nDets), dtype=np.float32)

    # Detector edges:
    Dets_bnd = dDet * (np.arange(nDets+1) - nDets/2.0)

    # Image edges (centered)
    X_bnd = dPix * (np.arange(nX+1) - nX/2.0)
    Y_bnd = dPix * (np.arange(nY+1) - nY/2.0)

    for iAng, angle in enumerate(Angs):
        Angs_cos, Angs_sin = np.cos(angle), np.sin(angle)

        for iX, (xmin, xmax) in enumerate(zip(X_bnd[:-1], X_bnd[1:])):
            for iY, (ymin, ymax) in enumerate(zip(Y_bnd[:-1], Y_bnd[1:])):

                if img[iX, iY] == 0:
                    continue

                # Project pixel corners to detector axis
                # These are the min/max of axis_u . (corner position)
                corners = [Angs_cos*ymin - Angs_sin*xmin, Angs_cos*ymax - Angs_sin*xmin,
                           Angs_cos*ymin - Angs_sin*xmax, Angs_cos*ymax - Angs_sin*xmax]

                P_min = min(corners)
                P_max = max(corners)

                # Footprint: find detector bins that overlap with rectangle projection

                # Bin k: between det_edges[k], det_edges[k+1]
                iDet0 = np.searchsorted(Dets_bnd, P_min, side='right') - 1
                iDetN = np.searchsorted(Dets_bnd, P_max, side='left')

                # For each overlapped bin, calculate geometric overlap
                for iDet in range(max(0, iDet0), min(nDets, iDetN)):
                    left = max(P_min, Dets_bnd[iDet])
                    right = min(P_max, Dets_bnd[iDet+1])
                    if right > left:
                        # Normalize by bin width
                        sino[iAng, iDet] += img[iX, iY] * (right - left) / dDet

    return sino


"""

img = np.zeros((32, 32))
img[4:8, 4:8] = 1.0  # center impulse

angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
#angles = np.array([np.pi/4])
nDets = 64
dDet = .5

sino1 = siddons_fp_2d(img, angles, nDets=nDets,dDet=dDet)
sino2 = joseph_fp_2d(img, angles, nDets=nDets,dDet=dDet)
sino3 = distance_driven_fp_2d(img, angles, nDets=nDets,dDet=dDet)
sino4 = separable_footprint_fp_2d(img, angles, nDets,dDet=dDet)
print(sino1)
print(sino2)
print(sino3)
print(sino4)

plt.figure(figsize=(6,6))
plt.subplot(2,2,1)
plt.imshow(sino1, cmap='gray', aspect='auto')
plt.title("Siddons")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,2,2)
plt.imshow(sino2, cmap='gray', aspect='auto')
plt.title("Joseph")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,2,3)
plt.imshow(sino3, cmap='gray', aspect='auto')
plt.title("Distance-Driven")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,2,4)
plt.imshow(sino4, cmap='gray', aspect='auto')
plt.title("Seperable Footprint")
plt.xlabel("Detector bin")
plt.ylabel("Angle")
plt.tight_layout()
plt.show()



dDet = .5
nDets = 64
nAngs = 32
r = 12.5

Dets = dDet*(np.arange(nDets) - nDets / 2.0 + 0.5)


proj = 2*np.sqrt((r**2 - Dets**2).clip(0))
sino = np.tile(proj, [nAngs, 1])

Angs = np.linspace(0, np.pi*2, 32, endpoint=False)
nX = 32
nY = 32

rec = distance_driven_bp_2d(sino, Angs, nX, nY, dPix=1.0, dDet=dDet)

# plt.imshow(rec)

plt.plot(rec[:, 16])
plt.plot(rec[16, :])
plt.show()

plt.plot(rec[:, 16])
plt.plot(rec[:, 15])
plt.show()
"""