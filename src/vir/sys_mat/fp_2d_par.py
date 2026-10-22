# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 08:57:56 2026

@author: varga
"""

import numpy as np


def siddons_fp_2d(img, Angs, n_dets, d_det=1.0, d_pix=1.0):
    nX, nY = img.shape
    sino = np.zeros((Angs.size, n_dets), dtype=np.float32)

    # Image edges (centered)
    X_bnd = d_pix * (np.arange(nX+1) - nX/2.0)
    Y_bnd = d_pix * (np.arange(nY+1) - nY/2.0)

    # Detector bins (centered)
    Dets_cnt = d_det * (np.arange(n_dets) - n_dets/2.0 + 0.5)

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
                ix = int(np.floor((xc - X_bnd[0]) / d_pix))
                iy = int(np.floor((yc - Y_bnd[0]) / d_pix))
                if 0 <= ix < nX and 0 <= iy < nY:
                    sino[iAng, iDet] += img[ix, iy] * abs(tc1-tc0)
    return sino


def separable_footprint_fp_2d(img, Angs, n_dets, d_det=1.0, d_pix=1.0):
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
    nX, nY = img.shape
    sino = np.zeros((Angs.size, n_dets), dtype=np.float32)

    # Detector edges:
    Dets_bnd = d_det * (np.arange(n_dets+1) - n_dets/2.0)

    # Image edges (centered)
    X_bnd = d_pix * (np.arange(nX+1) - nX/2.0)
    Y_bnd = d_pix * (np.arange(nY+1) - nY/2.0)

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
                for iDet in range(max(0, iDet0), min(n_dets, iDetN)):
                    left = max(P_min, Dets_bnd[iDet])
                    right = min(P_max, Dets_bnd[iDet+1])
                    if right > left:
                        # Normalize by bin width
                        sino[iAng, iDet] += img[iX, iY] * (right - left) / d_det

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