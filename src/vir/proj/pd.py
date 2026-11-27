#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pixel-driven separable-footprint projectors (SF-TR: trapezoid in-plane
footprint whose corners are the four projected pixel corners, rectangle in v).

Footprint amplitude is set so its area equals the voxel's integrated line
integral:  amp = d_pix*M / (cos(psi)*W_eff),  W_eff = trapezoid area at unit
height, M = DSD/(DSO - o) (1 for parallel), psi = ray angle through the voxel
centre.  For 2D parallel beam the projected square is exactly a trapezoid,
so the model is exact there.

See vir.sys_mat.pf for geometry, scaling and naming conventions.

@author: varga
"""

import numpy as np
import vir.proj.pf as pf
from vir.proj.pf import njit


@njit(fastmath=True, inline='always', cache=True)
def _proj_u(x, y, cos_ang, sin_ang, DSO, DSD, fan):
    p = pf.rot_p(x, y, cos_ang, sin_ang)
    if fan:
        return pf.proj_fan(p, pf.rot_o(x, y, cos_ang, sin_ang), DSO, DSD)
    return p


@njit(fastmath=True, cache=True)
def _corner_u_grid(u_crn, x_bnd_arr, y_bnd_arr, cos_ang, sin_ang, DSO, DSD, fan):
    """u coordinate of every pixel corner, shape (nx+1, ny+1)."""
    for ix in range(x_bnd_arr.size):
        for iy in range(y_bnd_arr.size):
            u_crn[ix, iy] = _proj_u(x_bnd_arr[ix], y_bnd_arr[iy],
                                    cos_ang, sin_ang, DSO, DSD, fan)


@njit(fastmath=True, inline='always', cache=True)
def _footprint_u(u_crn, ix, iy):
    """Sorted trapezoid corners of pixel (ix, iy) and its unit-height area."""
    t0, t1, t2, t3 = pf.sort4(u_crn[ix, iy], u_crn[ix + 1, iy],
                              u_crn[ix, iy + 1], u_crn[ix + 1, iy + 1])
    return t0, t1, t2, t3, 0.5*(t3 + t2 - t1 - t0)


# ----------------------------------------------------------------------------
# 2D parallel / fan
# ----------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def pd_p_2d(img, sino, ang_arr, DSO, DSD, du, su, d_pix, fan, bp):
    """fp: sino += A img.  bp: img += A^T sino.  (in place)"""
    nx, ny = img.shape
    na, nu = sino.shape

    x_bnd_arr = pf.boundspace(nx, d_pix)
    y_bnd_arr = pf.boundspace(ny, d_pix)
    x_arr = pf.censpace(nx, d_pix)
    y_arr = pf.censpace(ny, d_pix)
    u_bnd_arr = pf.det_boundspace(nu, du, su)
    scl = d_pix/du

    u_crn = np.empty((nx + 1, ny + 1), dtype=np.float32)

    for ia in range(na):
        cos_ang = np.cos(ang_arr[ia])
        sin_ang = np.sin(ang_arr[ia])
        _corner_u_grid(u_crn, x_bnd_arr, y_bnd_arr, cos_ang, sin_ang, DSO, DSD, fan)

        for ix in range(nx):
            for iy in range(ny):
                if not bp and img[ix, iy] == 0:
                    continue

                t0, t1, t2, t3, w_eff = _footprint_u(u_crn, ix, iy)
                iu0, iu1 = pf.bin_range(u_bnd_arr, t0, t3)
                if iu0 >= iu1 or w_eff <= 0:
                    continue

                if fan:
                    p_c = pf.rot_p(x_arr[ix], y_arr[iy], cos_ang, sin_ang)
                    mag = DSD/(DSO - pf.rot_o(x_arr[ix], y_arr[iy], cos_ang, sin_ang))
                    amp = d_pix*mag/(pf.cos_psi(mag*p_c, 0.0, DSD)*w_eff)
                else:
                    amp = d_pix/w_eff

                if bp:
                    acc = 0.0
                    for iu in range(iu0, iu1):
                        acc += sino[ia, iu]*pf.trap_overlap(
                            u_bnd_arr[iu], u_bnd_arr[iu + 1], t0, t1, t2, t3)
                    img[ix, iy] += acc*amp*scl
                else:
                    val = img[ix, iy]*amp*scl
                    for iu in range(iu0, iu1):
                        sino[ia, iu] += val*pf.trap_overlap(
                            u_bnd_arr[iu], u_bnd_arr[iu + 1], t0, t1, t2, t3)


def _fp_2d(img, ang_arr, nu, DSO, DSD, du, su, d_pix, fan):
    img, ang_arr, du, su, d_pix = \
        pf.as_float32(img, ang_arr, du, su, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    if fan:
        pf.check_fov(img.shape, d_pix, DSO)
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)
    pd_p_2d(img, sino, ang_arr, DSO, DSD, du, su, d_pix, fan, False)
    return sino


def _bp_2d(sino, ang_arr, img_shape, DSO, DSD, du, su, d_pix, fan, norm):
    sino, ang_arr, du, su, d_pix = \
        pf.as_float32(sino, ang_arr, du, su, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    if fan:
        pf.check_fov(img_shape, d_pix, DSO)
    img = np.zeros(img_shape, dtype=np.float32)
    pd_p_2d(img, sino, ang_arr, DSO, DSD, du, su, d_pix, fan, True)
    if norm:
        img *= pf.bp_norm(ang_arr.size, d_pix, du)
    return img


def pd_fp_par_2d(img, ang_arr, nu, du=1.0, su=0.0, d_pix=1.0):
    return _fp_2d(img, ang_arr, nu, 1.0, 1.0, du, su, d_pix, False)


def pd_bp_par_2d(sino, ang_arr, img_shape, du=1.0, su=0.0, d_pix=1.0, norm=True):
    return _bp_2d(sino, ang_arr, img_shape, 1.0, 1.0, du, su, d_pix, False, norm)


def pd_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=1.0, su=0.0, d_pix=1.0):
    return _fp_2d(img, ang_arr, nu, DSO, DSD, du, su, d_pix, True)


def pd_bp_fan_2d(sino, ang_arr, img_shape, DSO, DSD, du=1.0, su=0.0,
                 d_pix=1.0, norm=True):
    return _bp_2d(sino, ang_arr, img_shape, DSO, DSD, du, su, d_pix, True, norm)


# ----------------------------------------------------------------------------
# 3D circular cone beam
# ----------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def pd_p_cone_3d(img, sino, ang_arr, DSO, DSD, du, dv, su, sv, d_pix, bp):
    nx, ny, nz = img.shape
    na, nu, nv = sino.shape

    x_bnd_arr = pf.boundspace(nx, d_pix)
    y_bnd_arr = pf.boundspace(ny, d_pix)
    z_bnd_arr = pf.boundspace(nz, d_pix)
    x_arr = pf.censpace(nx, d_pix)
    y_arr = pf.censpace(ny, d_pix)
    z_arr = pf.censpace(nz, d_pix)
    u_bnd_arr = pf.det_boundspace(nu, du, su)
    v_bnd_arr = pf.det_boundspace(nv, dv, sv)
    scl = d_pix/(du*dv)

    u_crn = np.empty((nx + 1, ny + 1), dtype=np.float32)

    for ia in range(na):
        cos_ang = np.cos(ang_arr[ia])
        sin_ang = np.sin(ang_arr[ia])
        _corner_u_grid(u_crn, x_bnd_arr, y_bnd_arr, cos_ang, sin_ang, DSO, DSD, True)

        for ix in range(nx):
            for iy in range(ny):
                t0, t1, t2, t3, w_eff = _footprint_u(u_crn, ix, iy)
                iu0, iu1 = pf.bin_range(u_bnd_arr, t0, t3)
                if iu0 >= iu1 or w_eff <= 0:
                    continue

                # Flat detector: v = M*z with M at the voxel-column centre
                p_c = pf.rot_p(x_arr[ix], y_arr[iy], cos_ang, sin_ang)
                o_c = pf.rot_o(x_arr[ix], y_arr[iy], cos_ang, sin_ang)
                mag = DSD/(DSO - o_c)
                u_c = mag*p_c

                for iz in range(nz):
                    if not bp and img[ix, iy, iz] == 0:
                        continue
                    v_min = mag*z_bnd_arr[iz]
                    v_max = mag*z_bnd_arr[iz + 1]
                    iv0, iv1 = pf.bin_range(v_bnd_arr, v_min, v_max)
                    if iv0 >= iv1:
                        continue
                    amp = d_pix*mag/(pf.cos_psi(u_c, mag*z_arr[iz], DSD)*w_eff)

                    if bp:
                        acc = 0.0
                        for iu in range(iu0, iu1):
                            ov_u = pf.trap_overlap(u_bnd_arr[iu], u_bnd_arr[iu + 1],
                                                   t0, t1, t2, t3)
                            for iv in range(iv0, iv1):
                                acc += sino[ia, iu, iv]*ov_u*pf.overlap(
                                    v_min, v_max, v_bnd_arr[iv], v_bnd_arr[iv + 1])
                        img[ix, iy, iz] += acc*amp*scl
                    else:
                        val = img[ix, iy, iz]*amp*scl
                        for iu in range(iu0, iu1):
                            ov_u = val*pf.trap_overlap(u_bnd_arr[iu], u_bnd_arr[iu + 1],
                                                       t0, t1, t2, t3)
                            for iv in range(iv0, iv1):
                                sino[ia, iu, iv] += ov_u*pf.overlap(
                                    v_min, v_max, v_bnd_arr[iv], v_bnd_arr[iv + 1])


def pd_fp_cone_3d(img, ang_arr, nu, nv, DSO, DSD, du=1.0, dv=1.0, su=0.0,
                  sv=0.0, d_pix=1.0):
    img, ang_arr, du, dv, su, sv, d_pix = \
        pf.as_float32(img, ang_arr, du, dv, su, sv, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    pf.check_fov(img.shape, d_pix, DSO)
    sino = np.zeros((ang_arr.size, nu, nv), dtype=np.float32)
    pd_p_cone_3d(img, sino, ang_arr, DSO, DSD, du, dv, su, sv, d_pix, False)
    return sino


def pd_bp_cone_3d(sino, ang_arr, img_shape, DSO, DSD, du=1.0, dv=1.0, su=0.0,
                  sv=0.0, d_pix=1.0, norm=True):
    sino, ang_arr, du, dv, su, sv, d_pix = \
        pf.as_float32(sino, ang_arr, du, dv, su, sv, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    pf.check_fov(img_shape, d_pix, DSO)
    img = np.zeros(img_shape, dtype=np.float32)
    pd_p_cone_3d(img, sino, ang_arr, DSO, DSD, du, dv, su, sv, d_pix, True)
    if norm:
        img *= pf.bp_norm(ang_arr.size, d_pix, du, dv, 3)
    return img
