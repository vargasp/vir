#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 06:34:23 2026

@author: pvargas21
"""

import numpy as np

def distance_driven_fp_fan_2d(img, Angs, nDets, R, D,
                              dPix=1.0, dDet=1.0):
    """
    Distance-driven fan-beam forward projector (2D).

    Parameters:
        img   : (nX, nY)
        Angs  : projection angles (rad)
        nDets : number of detector bins
        R     : source-to-isocenter distance
        D     : source-to-detector distance
    """
    nX, nY = img.shape
    sino = np.zeros((len(Angs), nDets), dtype=np.float32)

    # Pixel boundaries
    X_bnd = dPix * (np.arange(nX + 1) - nX / 2)
    Y_bnd = dPix * (np.arange(nY + 1) - nY / 2)
    X_cnt = 0.5 * (X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5 * (Y_bnd[:-1] + Y_bnd[1:])

    # Detector boundaries
    Dets_bnd = dDet * (np.arange(nDets + 1) - nDets / 2)

    for ia, ang in enumerate(Angs):
        cos_t, sin_t = np.cos(ang), np.sin(ang)

        # Source position
        xs, ys = R * cos_t, R * sin_t

        # Detector axis (unit vector)
        ux, uy = -sin_t, cos_t

        # Distance from source to detector line
        SD = D

        # Choose driving axis
        if abs(cos_t) >= abs(sin_t):
            # X-driven
            for iy in range(nY):
                y = Y_cnt[iy]

                for ix in range(nX):
                    # Pixel x boundaries
                    xb0, xb1 = X_bnd[ix], X_bnd[ix + 1]

                    # Project pixel edges to detector
                    for xb in (xb0, xb1):
                        dx = xb - xs
                        dy = y  - ys
                        l = SD / (dx * cos_t + dy * sin_t)
                        t = l * (dx * ux + dy * uy)
                        yield_t = t

                    t0, t1 = sorted([yield_t, t])

                    iP = ix
                    iDet = np.searchsorted(Dets_bnd, t0) - 1
                    iDet = max(iDet, 0)

                    while iDet < nDets and Dets_bnd[iDet] < t1:
                        left  = max(t0, Dets_bnd[iDet])
                        right = min(t1, Dets_bnd[iDet + 1])
                        if right > left:
                            sino[ia, iDet] += img[ix, iy] * (right - left)
                        iDet += 1

        else:
            # Y-driven
            for ix in range(nX):
                x = X_cnt[ix]

                for iy in range(nY):
                    yb0, yb1 = Y_bnd[iy], Y_bnd[iy + 1]

                    for yb in (yb0, yb1):
                        dx = x  - xs
                        dy = yb - ys
                        l = SD / (dx * cos_t + dy * sin_t)
                        t = l * (dx * ux + dy * uy)
                        yield_t = t

                    t0, t1 = sorted([yield_t, t])

                    iDet = np.searchsorted(Dets_bnd, t0) - 1
                    iDet = max(iDet, 0)

                    while iDet < nDets and Dets_bnd[iDet] < t1:
                        left  = max(t0, Dets_bnd[iDet])
                        right = min(t1, Dets_bnd[iDet + 1])
                        if right > left:
                            sino[ia, iDet] += img[ix, iy] * (right - left)
                        iDet += 1

    return sino


def distance_driven_bp_fan_2d(sino, Angs, nX, nY, R, D,
                              dPix=1.0, dDet=1.0):
    """
    Adjoint distance-driven fan-beam backprojector (2D).
    """
    img = np.zeros((nX, nY), dtype=np.float32)

    X_bnd = dPix * (np.arange(nX + 1) - nX / 2)
    Y_bnd = dPix * (np.arange(nY + 1) - nY / 2)
    X_cnt = 0.5 * (X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5 * (Y_bnd[:-1] + Y_bnd[1:])
    Dets_bnd = dDet * (np.arange(sino.shape[1] + 1) - sino.shape[1] / 2)

    for ia, ang in enumerate(Angs):
        cos_t, sin_t = np.cos(ang), np.sin(ang)
        xs, ys = R * cos_t, R * sin_t
        ux, uy = -sin_t, cos_t
        SD = D
        proj = sino[ia]

        if abs(cos_t) >= abs(sin_t):
            for iy in range(nY):
                y = Y_cnt[iy]
                for ix in range(nX):
                    xb0, xb1 = X_bnd[ix], X_bnd[ix + 1]

                    t_vals = []
                    for xb in (xb0, xb1):
                        dx = xb - xs
                        dy = y  - ys
                        l = SD / (dx * cos_t + dy * sin_t)
                        t_vals.append(l * (dx * ux + dy * uy))

                    t0, t1 = sorted(t_vals)
                    iDet = np.searchsorted(Dets_bnd, t0) - 1
                    iDet = max(iDet, 0)

                    while iDet < proj.size and Dets_bnd[iDet] < t1:
                        left  = max(t0, Dets_bnd[iDet])
                        right = min(t1, Dets_bnd[iDet + 1])
                        if right > left:
                            img[ix, iy] += proj[iDet] * (right - left)
                        iDet += 1
        else:
            for ix in range(nX):
                x = X_cnt[ix]
                for iy in range(nY):
                    yb0, yb1 = Y_bnd[iy], Y_bnd[iy + 1]

                    t_vals = []
                    for yb in (yb0, yb1):
                        dx = x  - xs
                        dy = yb - ys
                        l = SD / (dx * cos_t + dy * sin_t)
                        t_vals.append(l * (dx * ux + dy * uy))

                    t0, t1 = sorted(t_vals)
                    iDet = np.searchsorted(Dets_bnd, t0) - 1
                    iDet = max(iDet, 0)

                    while iDet < proj.size and Dets_bnd[iDet] < t1:
                        left  = max(t0, Dets_bnd[iDet])
                        right = min(t1, Dets_bnd[iDet + 1])
                        if right > left:
                            img[ix, iy] += proj[iDet] * (right - left)
                        iDet += 1

    return img


def distance_driven_fp_fan_arc_2d(img, Angs, nDets, R, D,
                                   dPix=1.0, dGamma=1e-3):
    """
    Distance-driven fan-beam forward projector (arc detector).
    """
    nX, nY = img.shape
    sino = np.zeros((len(Angs), nDets), dtype=np.float32)

    # Pixel boundaries
    X_bnd = dPix * (np.arange(nX + 1) - nX / 2)
    Y_bnd = dPix * (np.arange(nY + 1) - nY / 2)
    X_cnt = 0.5 * (X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5 * (Y_bnd[:-1] + Y_bnd[1:])

    # Detector angular boundaries
    Gam_bnd = dGamma * (np.arange(nDets + 1) - nDets / 2)

    for ia, ang in enumerate(Angs):
        cos_t, sin_t = np.cos(ang), np.sin(ang)

        xs, ys = R * cos_t, R * sin_t
        erx, ery = cos_t, sin_t
        etx, ety = -sin_t, cos_t

        # Choose driving axis
        if abs(cos_t) >= abs(sin_t):  # X-driven
            for iy in range(nY):
                y = Y_cnt[iy]

                for ix in range(nX):
                    gam = []

                    for xb in (X_bnd[ix], X_bnd[ix + 1]):
                        dx = xb - xs
                        dy = y  - ys
                        u = dx * etx + dy * ety
                        v = dx * erx + dy * ery
                        gam.append(np.arctan2(u, v))

                    g0, g1 = sorted(gam)

                    iDet = np.searchsorted(Gam_bnd, g0) - 1
                    iDet = max(iDet, 0)

                    while iDet < nDets and Gam_bnd[iDet] < g1:
                        left  = max(g0, Gam_bnd[iDet])
                        right = min(g1, Gam_bnd[iDet + 1])
                        if right > left:
                            sino[ia, iDet] += img[ix, iy] * (right - left)
                        iDet += 1
        else:  # Y-driven
            for ix in range(nX):
                x = X_cnt[ix]

                for iy in range(nY):
                    gam = []

                    for yb in (Y_bnd[iy], Y_bnd[iy + 1]):
                        dx = x  - xs
                        dy = yb - ys
                        u = dx * etx + dy * ety
                        v = dx * erx + dy * ery
                        gam.append(np.arctan2(u, v))

                    g0, g1 = sorted(gam)

                    iDet = np.searchsorted(Gam_bnd, g0) - 1
                    iDet = max(iDet, 0)

                    while iDet < nDets and Gam_bnd[iDet] < g1:
                        left  = max(g0, Gam_bnd[iDet])
                        right = min(g1, Gam_bnd[iDet + 1])
                        if right > left:
                            sino[ia, iDet] += img[ix, iy] * (right - left)
                        iDet += 1

    return sino


def distance_driven_bp_fan_arc_2d(sino, Angs, nX, nY, R, D,
                                   dPix=1.0, dGamma=1e-3):
    """
    Adjoint distance-driven fan-beam backprojection (arc detector).
    """
    img = np.zeros((nX, nY), dtype=np.float32)
    nDets = sino.shape[1]

    X_bnd = dPix * (np.arange(nX + 1) - nX / 2)
    Y_bnd = dPix * (np.arange(nY + 1) - nY / 2)
    X_cnt = 0.5 * (X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5 * (Y_bnd[:-1] + Y_bnd[1:])
    Gam_bnd = dGamma * (np.arange(nDets + 1) - nDets / 2)

    for ia, ang in enumerate(Angs):
        cos_t, sin_t = np.cos(ang), np.sin(ang)
        xs, ys = R * cos_t, R * sin_t
        erx, ery = cos_t, sin_t
        etx, ety = -sin_t, cos_t
        proj = sino[ia]

        if abs(cos_t) >= abs(sin_t):
            for iy in range(nY):
                y = Y_cnt[iy]

                for ix in range(nX):
                    gam = []
                    for xb in (X_bnd[ix], X_bnd[ix + 1]):
                        dx = xb - xs
                        dy = y  - ys
                        u = dx * etx + dy * ety
                        v = dx * erx + dy * ery
                        gam.append(np.arctan2(u, v))

                    g0, g1 = sorted(gam)

                    iDet = np.searchsorted(Gam_bnd, g0) - 1
                    iDet = max(iDet, 0)

                    while iDet < nDets and Gam_bnd[iDet] < g1:
                        left  = max(g0, Gam_bnd[iDet])
                        right = min(g1, Gam_bnd[iDet + 1])
                        if right > left:
                            img[ix, iy] += proj[iDet] * (right - left)
                        iDet += 1
        else:
            for ix in range(nX):
                x = X_cnt[ix]

                for iy in range(nY):
                    gam = []
                    for yb in (Y_bnd[iy], Y_bnd[iy + 1]):
                        dx = x  - xs
                        dy = yb - ys
                        u = dx * etx + dy * ety
                        v = dx * erx + dy * ery
                        gam.append(np.arctan2(u, v))

                    g0, g1 = sorted(gam)

                    iDet = np.searchsorted(Gam_bnd, g0) - 1
                    iDet = max(iDet, 0)

                    while iDet < nDets and Gam_bnd[iDet] < g1:
                        left  = max(g0, Gam_bnd[iDet])
                        right = min(g1, Gam_bnd[iDet + 1])
                        if right > left:
                            img[ix, iy] += proj[iDet] * (right - left)
                        iDet += 1

    return img