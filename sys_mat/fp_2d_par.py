# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 08:57:56 2026

@author: varga
"""


import numpy as np


def _dd_fp_2d_sweep(sino, img, drive_axis, offset, Dets_bnd, nP, nP2, inv, iAng):

    nDets = Dets_bnd.size - 1
    
    for iP2 in range(nP2):
        
        # Projected pixel boundaries at the detector plane
        P_bnd = drive_axis + offset[iP2]
        img_vec = img[:, iP2]

        iP = iDet = 0
        while iP < nP and iDet < nDets:
            left = max(P_bnd[iP],   Dets_bnd[iDet])
            right = min(P_bnd[iP+1], Dets_bnd[iDet+1])
            if right > left:
                sino[iAng, iDet] += img_vec[iP] * (right-left) / inv

            # Advance to next pixel or detector
            if P_bnd[iP+1] < Dets_bnd[iDet+1]:
                iP += 1
            else:
                iDet += 1







def dd_fp_2d(img, Angs, nDets, dPix=1.0, dDet=1.0):
    """
    Distance-driven forward projection for 2D parallel-beam CT.

    Parameters:
        img     : ndarray shape (nX, nY)
        Angs    : ndarray of projection Angs (radians)
        nDets   : number of detector bins
        dPix    : pixel width
        dDet    : detector bin width
    Returns:
        sino    : ndarray shape (len(angles), nDets) -- the sinogram
    """
    nX, nY = img.shape
    sino = np.zeros((Angs.size, nDets), dtype=np.float32)

    # Pixel boundaries coordinates
    X_bnd = dPix * (np.arange(nX+1) - nX/2.0)
    Y_bnd = dPix * (np.arange(nY+1) - nY/2.0)

    # Pixel center coordinates
    X_cnt = 0.5 * (X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5 * (Y_bnd[:-1] + Y_bnd[1:])

    # Detector coordinates
    Dets_bnd = dDet * (np.arange(nDets+1) - nDets/2.0)

    for iAng, angle in enumerate(Angs):
        ang_cos, ang_sin = np.cos(angle), np.sin(angle)

        # X-driven: projection axis is closer to X
        if abs(ang_cos) >= abs(ang_sin):
            nP1, nP2 = nX, nY
            inv = max(dDet*abs(ang_cos), 1e-12)

            # Rotate the driving axis
            drive_axis = ang_cos * X_bnd
            offset = ang_sin * -Y_cnt

            # If X_cos is decreasing, create reverse views of relevant axes
            if drive_axis[1] < drive_axis[0]:
                drive_axis = drive_axis[::-1]
                imgP = img[::-1, :]
            else:
                imgP = img

        # Y-driven: projection axis is closer to Y
        else:
            nP1, nP2 = nY, nX
            inv = max(dDet*abs(ang_sin), 1e-12)
            drive_axis = ang_sin * -Y_bnd
            offset = ang_cos * X_cnt

            # If drive_axis is decreasing, create reverse views of relevant axes
            if drive_axis[1] < drive_axis[0]:
                drive_axis = drive_axis[::-1]
                imgP = img[:, ::-1].T
            else:
                imgP = img.T

        _dd_fp_2d_sweep(sino, imgP, drive_axis, offset,
                        Dets_bnd, nP1, nP2, inv, iAng)

    return sino




def dd_fp_fan_2d(img, Angs, nDets, DSO, DSD, dPix=1.0, dDet=1.0):
    """
    Distance-driven fan-beam forward projector with full 360° coverage.
    Handles negative slopes and rays behind the source.

    Parameters
    ----------
    img     : ndarray shape (nX, nY)
    Angs    : ndarray of projection angles (radians)
    nDets   : number of detector bins
    DSO     : distance from source to origin
    DSD     : distance from source to detector
    dPix    : pixel width
    dDet    : detector bin width

    Returns
    -------
    sino    : ndarray shape (len(Angs), nDets)
    """
    nX, nY = img.shape
    sino = np.zeros((len(Angs), nDets), dtype=np.float32)

    # Pixel boundaries and centers
    X_bnd = dPix*(np.arange(nX+1) - nX/2)
    Y_bnd = dPix*(np.arange(nY+1) - nY/2)
    X_cnt = 0.5*(X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5*(Y_bnd[:-1] + Y_bnd[1:])

    # Detector boundaries
    Dets_bnd = dDet*(np.arange(nDets+1) - nDets/2)

    for iAng, ang in enumerate(Angs):
        cos_t, sin_t = np.cos(ang), np.sin(ang)

        # Determine driving axis
        if abs(cos_t) >= abs(sin_t):
            # X-driven
            nP, nP2 = nX, nY
            drive_axis = X_bnd
            offset     = -Y_cnt
            imgP       = img
            swap_xy = False
            
        else:
            # Y-driven
            nP, nP2 = nY, nX
            drive_axis = -Y_bnd
            offset     = X_cnt
            imgP       = img.T
            swap_xy = True

        #Precompute P_bnd_ref for iP2=0
        if swap_xy:
            x = np.full(nP+1, offset[0])
            y = drive_axis
        else:
            x = drive_axis
            y = np.full(nP+1, offset[0])
        
        x_rot = cos_t*x + sin_t*y
        y_rot = -sin_t*x + cos_t*y
        P_bnd_ref = DSD * x_rot / (DSO - y_rot)
        flip_sweep = P_bnd_ref[-1] < P_bnd_ref[0]


        c_drive_axis = cos_t * drive_axis
        s_drive_axis = sin_t * drive_axis
        c_off_axis = cos_t * offset
        s_off_axis = sin_t * offset

        # Loop over orthogonal pixels
        for iP2 in range(nP2):
            img_vec = imgP[:, iP2]

            
            #o = np.full(nP+1, offset[iP2])
            if swap_xy:
                x_rot = s_drive_axis + c_off_axis[iP2]
                y_rot = c_drive_axis - s_off_axis[iP2]
            else:
                x_rot = c_drive_axis + s_off_axis[iP2]
                y_rot = -s_drive_axis + c_off_axis[iP2]

            
            # Compute denominators
            denom = DSO - y_rot
                        
            # Compute projected boundaries
            P_bnd = DSD * x_rot / denom
            
            # Detect overall negative slope and flip sweep if needed
            if flip_sweep:
                P_bnd = P_bnd[::-1]
                img_vec = img_vec[::-1]

            # Distance-driven sweep
            iP = iDet = 0
            while iP < nP and iDet < nDets:
                left  = max(P_bnd[iP],     Dets_bnd[iDet])
                right = min(P_bnd[iP+1],   Dets_bnd[iDet+1])
                if right > left:
                    sino[iAng, iDet] += img_vec[iP] * (right - left)/DSD

                
                # Advance to next pixel or detector                
                if P_bnd[iP+1] < Dets_bnd[iDet+1]:
                    iP += 1
                else:
                    iDet += 1

    return sino







def _dd_bp_2d_sweep(img,proj,drive_axis,offset,det_bnd,nP,nP2,inv,flip,swap_xy):
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

    if inv < 1e-12:
        inv = 1e-12

    n_dets = det_bnd.size - 1

    for i_orth in range(nP2):
        P_bnd = drive_axis + offset[i_orth]

        # Early rejection if fully off detector
        if P_bnd[-1] <= det_bnd[0] or P_bnd[0] >= det_bnd[-1]:
            continue

        i_pix = 0
        i_det = 0

        while i_pix < nP and i_det < n_dets:
            left  = P_bnd[i_pix]     if P_bnd[i_pix]     > det_bnd[i_det]     else det_bnd[i_det]
            right = P_bnd[i_pix + 1] if P_bnd[i_pix + 1] < det_bnd[i_det + 1] else det_bnd[i_det + 1]

            if right > left:
                val = proj[i_det] * (right - left) / inv

                ip = nP - 1 - i_pix if flip else i_pix

                if swap_xy:
                    img[i_orth, ip] += val
                else:
                    img[ip, i_orth] += val
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

            if drive_axis[1] < drive_axis[0]:
                drive_axis = drive_axis[::-1]
                flip_ax = True
            else:
                flip_ax = False

            swap_xy = False
            
        # Y-driven
        else:
            nP, nP2 = nY, nX
            inv = max(dDet*abs(ang_sin), 1e-12)
            drive_axis = ang_sin * -Y_bnd
            offset = ang_cos * X_cnt

            if drive_axis[1] < drive_axis[0]:
                drive_axis = drive_axis[::-1]
                flip_ax = True
            else:
                flip_ax = False
            
            swap_xy = True
            
        _dd_bp_2d_sweep(img,proj,drive_axis,offset,Dets_bnd,nP,nP2,inv,flip_ax,swap_xy)
            
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
                x_rot = cos_t * orth_cnt[i_orth] + sin_t * s
            else:
                x_rot = cos_t * s + sin_t * orth_cnt[i_orth]

            u_bnd[i] = DSD * x_rot / denom

        i_pix = 0
        i_det = 0

        while i_pix < n_sweep and i_det < n_det:
            left  = max(u_bnd[i_pix],     det_bnd[i_det])
            right = min(u_bnd[i_pix + 1], det_bnd[i_det + 1])

            if right > left:
                val = proj[i_det] * (right - left)
                if swap_xy:
                    img[i_orth, i_pix] += val
                else:
                    img[i_pix, i_orth] += val

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