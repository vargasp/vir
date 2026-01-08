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


def _dd_fp_fan_sweep(sino,img_rot,x_rot_bnd_drive,y_rot_bnd_drive,
                     x_rot_cnt_orth,y_rot_cnt_orth,u_det_bnd,
                     n_pix_drive,n_pix_orth,DSO,DSD,i_ang):
    """
    Distance-driven sweep kernel for 2D fan-beam forward projection.

    This function performs the core distance-driven accumulation for a single
    fan-beam projection angle. Pixel boundaries are rotated into the scanner
    coordinate system and then projected onto the detector via perspective
    geometry.

    Geometry:
        - Fan-beam
        - Point source at distance DSO from origin
        - Flat detector at distance DSD from source

    Parameters
    ----------
    sino : ndarray, shape (n_angles, n_det)
        Output sinogram array to be accumulated in-place.
    img_rot : ndarray, shape (n_pix_drive, n_pix_orth)
        Image data after transpose and/or flip so that axis 0 corresponds
        to the driving (sweep) axis.
    x_rot_bnd_drive : ndarray, shape (n_pix_drive + 1,)
        Rotated x-coordinates of pixel boundaries along the driving axis,
        independent of orthogonal pixel index.
    y_rot_bnd_drive : ndarray, shape (n_pix_drive + 1,)
        Rotated y-coordinates of pixel boundaries along the driving axis,
        independent of orthogonal pixel index.
    x_rot_cnt_orth : ndarray, shape (n_pix_orth,)
        Orthogonal pixel-center x-offsets after rotation.
    y_rot_cnt_orth : ndarray, shape (n_pix_orth,)
        Orthogonal pixel-center y-offsets after rotation.
    u_det_bnd : ndarray, shape (n_det + 1,)
        Detector bin boundary coordinates in detector space.
    n_pix_drive : int
        Number of pixels along the driving axis.
    n_pix_orth : int
        Number of pixels along the orthogonal axis.
    DSO : float
        Distance from source to origin.
    DSD : float
        Distance from source to detector.
    i_ang : int
        Index of the current projection angle in the sinogram.

    Notes
    -----
    - Pixel boundaries are projected using u = DSD * x / (DSO - y).
    - The sweep assumes projected pixel boundaries are monotonic.
    - Rays behind the source are not explicitly filtered.
    - Accumulation is performed in-place.
    """

    n_det = u_det_bnd.size - 1

    for i_orth in range(n_pix_orth):
        pix_line = img_rot[:, i_orth]

        # Rotate pixel boundary coordinates for this pixel line
        x_rot = x_rot_bnd_drive + x_rot_cnt_orth[i_orth]
        y_rot = y_rot_bnd_drive + y_rot_cnt_orth[i_orth]

        # Project pixel boundaries onto detector coordinate u
        # u = DSD * x / (DSO - y)
        u_pix_bnd = DSD * x_rot / (DSO - y_rot)

        i_pix = i_det = 0
        while i_pix < n_pix_drive and i_det < n_det:

            left  = max(u_pix_bnd[i_pix],   u_det_bnd[i_det])
            right = min(u_pix_bnd[i_pix+1], u_det_bnd[i_det+1])

            if right > left:
                #u_pix_bnd already includes fan-beam magnification; no additional DSD scaling here
                sino[i_ang, i_det] += pix_line[i_pix] * (right - left)

            if u_pix_bnd[i_pix+1] < u_det_bnd[i_det+1]:
                i_pix += 1
            else:
                i_det += 1




def dd_fp_fan_2d(img, angles, n_det, DSO, DSD, d_pix=1.0, d_det=1.0):
    """
    Distance-driven forward projection for 2D fan-beam CT.

    This function computes a sinogram from a 2D image using a distance-driven
    formulation for fan-beam geometry with a flat detector. For each angle,
    a driving axis is selected to ensure monotonicity of projected pixel
    boundaries.

    Geometry:
        - Point source rotating around image origin
        - Flat detector opposite the source
        - Full 360-degree angular coverage

    Parameters
    ----------
    img : ndarray, shape (n_x, n_y)
        Input 2D image (attenuation coefficients).
    angles : ndarray, shape (n_angles,)
        Projection angles in radians.
    n_det : int
        Number of detector bins.
    DSO : float
        Distance from source to origin.
    DSD : float
        Distance from source to detector.
    d_pix : float, optional
        Pixel width (default is 1.0).
    d_det : float, optional
        Detector bin width (default is 1.0).

    Returns
    -------
    sino : ndarray, shape (n_angles, n_det)
        Computed fan-beam sinogram.

    Notes
    -----
    - Uses a flat detector parameterized by linear coordinate u.
    - The distance-driven method conserves projected mass but does not
      include explicit fan-beam Jacobian weighting.
    - Pixel and detector grids are assumed to be centered at zero.
    - Monotonicity of projected boundaries is enforced via axis flipping.
    """

    n_x, n_y = img.shape
    sino = np.zeros((len(angles), n_det), dtype=np.float32)

    # Pixel boundaries and centers
    x_bnd = d_pix * (np.arange(n_x + 1) - n_x / 2.0)
    y_bnd = d_pix * (np.arange(n_y + 1) - n_y / 2.0)
    x_cnt = 0.5 * (x_bnd[:-1] + x_bnd[1:])
    y_cnt = 0.5 * (y_bnd[:-1] + y_bnd[1:])

    # Detector bin boundaries
    u_det_bnd = d_det * (np.arange(n_det + 1) - n_det / 2.0)

    for i_ang, theta in enumerate(angles):
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        if abs(cos_t) >= abs(sin_t):
            # X-driven
            n_pix_drive, n_pix_orth = n_x, n_y

            x_rot_bnd_drive = cos_t * x_bnd
            y_rot_bnd_drive = -sin_t * x_bnd

            x_rot_cnt_orth = -sin_t * y_cnt
            y_rot_cnt_orth = -cos_t * y_cnt

            img_rot = img

        else:
            # Y-driven
            n_pix_drive, n_pix_orth = n_y, n_x

            x_rot_bnd_drive = -sin_t * y_bnd
            y_rot_bnd_drive = -cos_t * y_bnd

            x_rot_cnt_orth = cos_t * x_cnt
            y_rot_cnt_orth = -sin_t * x_cnt

            img_rot = img.T

        # Ensure projected boundaries are monotonic
        x_test = x_rot_bnd_drive + x_rot_cnt_orth[0]
        y_test = y_rot_bnd_drive + y_rot_cnt_orth[0]
        u_test = DSD * x_test / (DSO - y_test)

        if u_test[-1] < u_test[0]:
            x_rot_bnd_drive = x_rot_bnd_drive[::-1]
            y_rot_bnd_drive = y_rot_bnd_drive[::-1]
            img_rot = img_rot[::-1, :]

        _dd_fp_fan_sweep(sino,img_rot,x_rot_bnd_drive,y_rot_bnd_drive,
                         x_rot_cnt_orth,y_rot_cnt_orth,u_det_bnd,
                         n_pix_drive,n_pix_orth,DSO, DSD,i_ang)

    return sino







def _dd_bp_2d_sweep(img,proj,drive_axis,offset,det_bnd,nP,nP2,inv):
    """
    Distance-driven backprojection sweep.

    Parameters
    ----------
    img            : 2D ndarray (output image, updated in-place)
    proj           : 1D ndarray (single projection view)
    sweep_pix_bnd  : 1D ndarray (pixel boundaries along sweep axis)
    orth_offset    : 1D ndarray (projection offsets per orth pixel)
    det_bnd        : 1D ndarray (detector bin boundaries)
    n_sweep        : int (number of pixels along sweep axis)
    n_orth         : int (number of pixels along orth axis)
    inv            : float (normalization factor)
    flip           : bool (reverse sweep index)
    swap_xy        : bool (write as img[y, x] instead of img[x, y])
    """

    n_dets = det_bnd.size - 1

    for i_orth in range(nP2):
        P_bnd = drive_axis + offset[i_orth]

        # Early rejection if fully off detector
        if P_bnd[-1] <= det_bnd[0] or P_bnd[0] >= det_bnd[-1]:
            continue

        i_pix = i_det = 0
        while i_pix < nP and i_det < n_dets:
            left  = max(P_bnd[i_pix],     det_bnd[i_det])
            right = min(P_bnd[i_pix + 1], det_bnd[i_det + 1])

            if right > left:
                img[i_pix, i_orth] += proj[i_det] * (right - left) / inv
            
            # Advance to next pixel or detector
            if P_bnd[i_pix + 1] < det_bnd[i_det + 1]:
                i_pix += 1
            else:
                i_det += 1
                

def dd_bp_2d(sino, Angs, nX, nY, dPix=1.0, dDet=1.0):
    """
    Adjoint distance-driven backprojection for 2D parallel-beam CT.

    Parameters:
        sino    : ndarray shape (len(Angs), nDets)
        Angs    : ndarray of projection angles (radians)
        nX, nY  : image size
        dPix    : pixel width
        dDet    : detector bin width

    Returns:
        img     : ndarray shape (nX, nY)
    """
    
    sino = np.asarray(sino, dtype=np.float32)
    nAngs, nDets = sino.shape
    img = np.zeros((nX, nY), dtype=np.float32)

    # Pixel boundaries
    X_bnd = dPix * (np.arange(nX + 1) - nX / 2.0)
    Y_bnd = dPix * (np.arange(nY + 1) - nY / 2.0)

    # Pixel centers
    X_cnt = 0.5 * (X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5 * (Y_bnd[:-1] + Y_bnd[1:])

    # Detector boundaries
    Dets_bnd = dDet * (np.arange(nDets + 1) - nDets / 2.0)

    for iAng, angle in enumerate(Angs):
        ang_cos, ang_sin = np.cos(angle), np.sin(angle)
        proj = sino[iAng]

        # X-driven
        if abs(ang_cos) >= abs(ang_sin):
            nP, nP2 = nX, nY
            inv = max(dDet*abs(ang_cos), 1e-12)
            drive_axis = ang_cos * X_bnd
            offset = ang_sin * -Y_cnt

            img_rot = img
            
        # Y-driven
        else:
            nP, nP2 = nY, nX
            inv = max(dDet*abs(ang_sin), 1e-12)
            drive_axis = ang_sin * -Y_bnd
            offset = ang_cos * X_cnt

            img_rot = img.T 
            
            
        if drive_axis[1] < drive_axis[0]:
            drive_axis = drive_axis[::-1]
            img_rot = img_rot[::-1, :]

            
        _dd_bp_2d_sweep(img_rot,proj,drive_axis,offset,Dets_bnd,nP,nP2,inv)
            
    return img













def _dd_bp_fan_sweep(
    img, proj,
    sweep_bnd, orth_cnt,
    det_bnd,
    DSO, DSD,
    cos_t, sin_t,
    swap_xy
):
    n_sweep = sweep_bnd.size - 1
    n_orth  = orth_cnt.size
    n_det   = det_bnd.size - 1

    for i_orth in range(n_orth):
        o = orth_cnt[i_orth]

        y_rot = cos_t * o
        denom = DSO - y_rot
        
        if denom <= 1e-12:
            continue

        u_bnd = np.empty(n_sweep + 1)
        for i in range(n_sweep + 1):
            s = sweep_bnd[i]
            if swap_xy:
                img_rot = img.T 
                x_rot = cos_t * orth_cnt[i_orth] + sin_t * s
            else:
                x_rot = cos_t * s + sin_t * orth_cnt[i_orth]
                img_rot = img

            u_bnd[i] = DSD * x_rot / denom

        i_pix = i_det = 0
        while i_pix < n_sweep and i_det < n_det:
            left  = max(u_bnd[i_pix],     det_bnd[i_det])
            right = min(u_bnd[i_pix + 1], det_bnd[i_det + 1])

            if right > left:
                img_rot[i_pix, i_orth] += proj[i_det] * (right - left)
                
            if u_bnd[i_pix + 1] < det_bnd[i_det + 1]:
                i_pix += 1
            else:
                i_det += 1





def dd_bp_fan_2d(sino, Angs, nX, nY, DSO, DSD, dPix=1.0, dDet=1.0):
    img = np.zeros((nX, nY), dtype=np.float32)

    X_bnd = dPix * (np.arange(nX + 1) - nX / 2)
    Y_bnd = dPix * (np.arange(nY + 1) - nY / 2)
    X_cnt = 0.5 * (X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5 * (Y_bnd[:-1] + Y_bnd[1:])
    Dets_bnd = dDet * (np.arange(sino.shape[1] + 1) - sino.shape[1] / 2)

    for i_ang, ang in enumerate(Angs):
        cos_t = np.cos(ang)
        sin_t = np.sin(ang)

        if abs(cos_t) >= abs(sin_t):
            sweep_bnd = X_bnd
            orth_cnt  = Y_cnt
            swap_xy   = False
        else:
            sweep_bnd = Y_bnd
            orth_cnt  = X_cnt
            swap_xy   = True

        _dd_bp_fan_sweep(
            img,
            sino[i_ang],
            sweep_bnd, orth_cnt,
            Dets_bnd,
            DSO, DSD,
            cos_t, sin_t,
            swap_xy
        )

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