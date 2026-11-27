#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance-driven projectors.

Per view the image is swept along the "driving" axis (x if |sin| >= |cos|,
else y), one row of the orthogonal axis at a time.  Pixel boundaries of the
row are projected onto the detector and merged with the detector bin
boundaries (O(n_drv + nu) per row).  The path-length weight |r|/|r_orth| is
evaluated per detector bin from the actual ray direction and applied
outside the sweep, so the sweep itself is pure geometry.

Memory layout: rows along the driving axis are contiguous
    x-driven rows: img_xrow[iy, ix]      (3D: [iy, iz, ix])
    y-driven rows: img_yrow[ix, iy]      (3D: [ix, iz, iy])

See vir.sys_mat.pf for geometry, scaling and naming conventions.

@author: pvargas21
"""

import numpy as np
import vir.proj.pf as pf
from vir.proj.pf import njit, prange


# ----------------------------------------------------------------------------
# Sweeps (bp=False: gather row -> detector, bp=True: scatter detector -> row)
# ----------------------------------------------------------------------------
@njit(fastmath=True, inline='always', cache=True)
def _sweep_1d(row, u_row, u_bnd_arr, det_vec, bp):
    """row[n_drv] <-> det_vec[nu]; u_row (n_drv+1) and u_bnd_arr (nu+1)
    are both increasing."""
    n_drv = row.size
    nu = u_bnd_arr.size - 1
    # Skip straight to the first overlapping pair
    ip = max(np.searchsorted(u_row, u_bnd_arr[0], side='right') - 1, 0)
    iu = max(np.searchsorted(u_bnd_arr, u_row[0], side='right') - 1, 0)
    while ip < n_drv and iu < nu:
        p_r = u_row[ip + 1]
        u_r = u_bnd_arr[iu + 1]
        ov = min(p_r, u_r) - max(u_row[ip], u_bnd_arr[iu])
        if ov > 0:
            if bp:
                row[ip] += det_vec[iu]*ov
            else:
                det_vec[iu] += row[ip]*ov
        if p_r < u_r:
            ip += 1
        else:
            iu += 1


@njit(fastmath=True, inline='always', cache=True)
def _sweep_cone(row, u_row, mag_row, z_l, z_r, u_bnd_arr, v_bnd_arr,
                det_vu, bp):
    """As _sweep_1d, with each voxel's v extent [mag*z_l, mag*z_r] using its
    own magnification.  det_vu has shape (nv, nu)."""
    n_drv = row.size
    nu = u_bnd_arr.size - 1
    ip = max(np.searchsorted(u_row, u_bnd_arr[0], side='right') - 1, 0)
    iu = max(np.searchsorted(u_bnd_arr, u_row[0], side='right') - 1, 0)

    ip_v = -1                   # voxel for which the v range is cached
    v_l = np.float32(0.0)
    v_r = np.float32(0.0)
    iv0 = 0
    iv1 = 0
    while ip < n_drv and iu < nu:
        p_r = u_row[ip + 1]
        u_r = u_bnd_arr[iu + 1]
        ov_u = min(p_r, u_r) - max(u_row[ip], u_bnd_arr[iu])
        if ov_u > 0 and (bp or row[ip] != 0):
            if ip != ip_v:
                v_l = mag_row[ip]*z_l
                v_r = mag_row[ip]*z_r
                iv0, iv1 = pf.bin_range(v_bnd_arr, v_l, v_r)
                ip_v = ip
            for iv in range(iv0, iv1):
                w = ov_u*pf.overlap(v_l, v_r, v_bnd_arr[iv], v_bnd_arr[iv + 1])
                if bp:
                    row[ip] += det_vu[iv, iu]*w
                else:
                    det_vu[iv, iu] += row[ip]*w
        if p_r < u_r:
            ip += 1
        else:
            iu += 1


# ----------------------------------------------------------------------------
# Per-view geometry
# ----------------------------------------------------------------------------
@njit(fastmath=True, inline='always', cache=True)
def _view_geom(cos_ang, sin_ang, x_bnd_arr, y_bnd_arr, x_arr, y_arr, x_drv):
    """Rotated coordinates of driving-axis boundaries and orthogonal-axis
    centres:  p = p_drv_bnd[k] + p_orth[io],  o = o_drv_bnd[k] + o_orth[io]."""
    if x_drv:
        return (-sin_ang*x_bnd_arr, cos_ang*y_arr,
                cos_ang*x_bnd_arr, sin_ang*y_arr)
    return (cos_ang*y_bnd_arr, -sin_ang*x_arr,
            sin_ang*y_bnd_arr, cos_ang*x_arr)


@njit(fastmath=True, inline='always', cache=True)
def _project_row(u_row, p_drv_bnd_arr, p_orth, o_drv_bnd_arr, o_orth,
                 DSO, DSD, fan):
    for k in range(p_drv_bnd_arr.size):
        if fan:
            u_row[k] = pf.proj_fan(p_drv_bnd_arr[k] + p_orth,
                                   o_drv_bnd_arr[k] + o_orth, DSO, DSD)
        else:
            u_row[k] = p_drv_bnd_arr[k] + p_orth


# ----------------------------------------------------------------------------
# 2D parallel / fan
# ----------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def dd_p_2d(img_xrow, img_yrow, sino, ang_arr, DSO, DSD, du, su, d_pix,
            fan, bp):
    """fp: sino += A img.  bp: img_xrow/img_yrow += A^T sino.  (in place)"""
    ny, nx = img_xrow.shape
    na, nu = sino.shape

    x_bnd_arr = pf.boundspace(nx, d_pix)
    y_bnd_arr = pf.boundspace(ny, d_pix)
    x_arr = pf.censpace(nx, d_pix)
    y_arr = pf.censpace(ny, d_pix)
    u_bnd_arr = pf.det_boundspace(nu, du, su)
    u_arr = pf.det_censpace(nu, du, su)
    v_arr = np.zeros(1, dtype=np.float32)
    scl = d_pix/du

    u_row_buf = np.empty(max(nx, ny) + 1, dtype=np.float32)
    det_vec = np.empty(nu, dtype=np.float32)

    for ia in range(na):
        cos_ang = np.cos(ang_arr[ia])
        sin_ang = np.sin(ang_arr[ia])
        x_drv = abs(sin_ang) >= abs(cos_ang)
        img_rows = img_xrow if x_drv else img_yrow
        n_orth, n_drv = img_rows.shape

        p_drv_bnd_arr, p_orth_arr, o_drv_bnd_arr, o_orth_arr = \
            _view_geom(cos_ang, sin_ang, x_bnd_arr, y_bnd_arr, x_arr, y_arr, x_drv)
        w_arr = pf.dd_weight_arr(cos_ang, sin_ang, u_arr, v_arr, DSD, fan, x_drv)

        for iu in range(nu):
            det_vec[iu] = sino[ia, iu]*w_arr[iu, 0]*scl if bp else 0.0

        for io in range(n_orth):
            u_row = u_row_buf[:n_drv + 1]
            _project_row(u_row, p_drv_bnd_arr, p_orth_arr[io],
                         o_drv_bnd_arr, o_orth_arr[io], DSO, DSD, fan)
            row = img_rows[io]
            if u_row[n_drv] < u_row[0]:          # make boundaries increasing
                u_row = u_row[::-1]
                row = row[::-1]
            _sweep_1d(row, u_row, u_bnd_arr, det_vec, bp)

        if not bp:
            for iu in range(nu):
                sino[ia, iu] = det_vec[iu]*w_arr[iu, 0]*scl


def _rows_2d(img):
    return np.ascontiguousarray(img.T), np.ascontiguousarray(img)


def _fp_2d(img, ang_arr, nu, DSO, DSD, du, su, d_pix, fan):
    img, ang_arr, du, su, d_pix = \
        pf.as_float32(img, ang_arr, du, su, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    if fan:
        pf.check_fov(img.shape, d_pix, DSO)
    img_xrow, img_yrow = _rows_2d(img)
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)
    dd_p_2d(img_xrow, img_yrow, sino, ang_arr, DSO, DSD, du, su, d_pix, fan, False)
    return sino


def _bp_2d(sino, ang_arr, img_shape, DSO, DSD, du, su, d_pix, fan, norm):
    sino, ang_arr, du, su, d_pix = \
        pf.as_float32(sino, ang_arr, du, su, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    if fan:
        pf.check_fov(img_shape, d_pix, DSO)
    nx, ny = img_shape
    img_xrow = np.zeros((ny, nx), dtype=np.float32)
    img_yrow = np.zeros((nx, ny), dtype=np.float32)
    dd_p_2d(img_xrow, img_yrow, sino, ang_arr, DSO, DSD, du, su, d_pix, fan, True)
    img = img_xrow.T + img_yrow
    if norm:
        img *= pf.bp_norm(ang_arr.size, d_pix, du)
    return img


def dd_fp_par_2d(img, ang_arr, nu, du=1.0, su=0.0, d_pix=1.0):
    return _fp_2d(img, ang_arr, nu, 1.0, 1.0, du, su, d_pix, False)


def dd_bp_par_2d(sino, ang_arr, img_shape, du=1.0, su=0.0, d_pix=1.0, norm=True):
    return _bp_2d(sino, ang_arr, img_shape, 1.0, 1.0, du, su, d_pix, False, norm)


def dd_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=1.0, su=0.0, d_pix=1.0):
    return _fp_2d(img, ang_arr, nu, DSO, DSD, du, su, d_pix, True)


def dd_bp_fan_2d(sino, ang_arr, img_shape, DSO, DSD, du=1.0, su=0.0,
                 d_pix=1.0, norm=True):
    return _bp_2d(sino, ang_arr, img_shape, DSO, DSD, du, su, d_pix, True, norm)


# ----------------------------------------------------------------------------
# 3D circular cone beam
# ----------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def dd_p_cone_3d(img_xrow, img_yrow, sino, ang_arr, DSO, DSD, du, dv, su, sv,
                 d_pix, bp):
    ny, nz, nx = img_xrow.shape
    na, nu, nv = sino.shape

    x_bnd_arr = pf.boundspace(nx, d_pix)
    y_bnd_arr = pf.boundspace(ny, d_pix)
    z_bnd_arr = pf.boundspace(nz, d_pix)
    x_arr = pf.censpace(nx, d_pix)
    y_arr = pf.censpace(ny, d_pix)
    u_bnd_arr = pf.det_boundspace(nu, du, su)
    v_bnd_arr = pf.det_boundspace(nv, dv, sv)
    u_arr = pf.det_censpace(nu, du, su)
    v_arr = pf.det_censpace(nv, dv, sv)
    v_lo = v_bnd_arr[0]
    v_hi = v_bnd_arr[nv]
    scl = d_pix/(du*dv)

    n_max = max(nx, ny)
    u_row_buf = np.empty(n_max + 1, dtype=np.float32)
    mag_buf = np.empty(n_max, dtype=np.float32)
    det_vu = np.empty((nv, nu), dtype=np.float32)

    for ia in range(na):
        cos_ang = np.cos(ang_arr[ia])
        sin_ang = np.sin(ang_arr[ia])
        x_drv = abs(sin_ang) >= abs(cos_ang)
        img_rows = img_xrow if x_drv else img_yrow
        n_orth, _, n_drv = img_rows.shape

        p_drv_bnd_arr, p_orth_arr, o_drv_bnd_arr, o_orth_arr = \
            _view_geom(cos_ang, sin_ang, x_bnd_arr, y_bnd_arr, x_arr, y_arr, x_drv)
        w_arr = pf.dd_weight_arr(cos_ang, sin_ang, u_arr, v_arr, DSD, True, x_drv)

        for iv in range(nv):
            for iu in range(nu):
                det_vu[iv, iu] = sino[ia, iu, iv]*w_arr[iu, iv]*scl if bp else 0.0

        for io in range(n_orth):
            u_row = u_row_buf[:n_drv + 1]
            mag_row = mag_buf[:n_drv]
            _project_row(u_row, p_drv_bnd_arr, p_orth_arr[io],
                         o_drv_bnd_arr, o_orth_arr[io], DSO, DSD, True)
            for k in range(n_drv):
                o_c = 0.5*(o_drv_bnd_arr[k] + o_drv_bnd_arr[k + 1]) + o_orth_arr[io]
                mag_row[k] = DSD/(DSO - o_c)
            mag_min = mag_row.min()
            mag_max = mag_row.max()
            flip = u_row[n_drv] < u_row[0]
            if flip:
                u_row = u_row[::-1]
                mag_row = mag_row[::-1]

            for iz in range(nz):
                z_l = z_bnd_arr[iz]
                z_r = z_bnd_arr[iz + 1]
                # Skip slabs whose whole row misses the detector in v
                if max(z_r*mag_min, z_r*mag_max) <= v_lo or \
                   min(z_l*mag_min, z_l*mag_max) >= v_hi:
                    continue
                row = img_rows[io, iz]
                if flip:
                    row = row[::-1]
                _sweep_cone(row, u_row, mag_row, z_l, z_r, u_bnd_arr,
                            v_bnd_arr, det_vu, bp)

        if not bp:
            for iu in range(nu):
                for iv in range(nv):
                    sino[ia, iu, iv] = det_vu[iv, iu]*w_arr[iu, iv]*scl


def _rows_3d(img):
    return (np.ascontiguousarray(img.transpose(1, 2, 0)),
            np.ascontiguousarray(img.transpose(0, 2, 1)))


def dd_fp_cone_3d(img, ang_arr, nu, nv, DSO, DSD, du=1.0, dv=1.0, su=0.0,
                  sv=0.0, d_pix=1.0, sample=1):
    img, ang_arr, du, dv, su, sv, d_pix = \
        pf.as_float32(img, ang_arr, du, dv, su, sv, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    if sample != 1:
        img = pf.upsample(img, sample)
        d_pix = np.float32(d_pix/sample)
    pf.check_fov(img.shape, d_pix, DSO)
    img_xrow, img_yrow = _rows_3d(img)
    sino = np.zeros((ang_arr.size, nu, nv), dtype=np.float32)
    dd_p_cone_3d(img_xrow, img_yrow, sino, ang_arr, DSO, DSD, du, dv, su, sv,
                 d_pix, False)
    return sino


def dd_bp_cone_3d(sino, ang_arr, img_shape, DSO, DSD, du=1.0, dv=1.0, su=0.0,
                  sv=0.0, d_pix=1.0, norm=True):
    sino, ang_arr, du, dv, su, sv, d_pix = \
        pf.as_float32(sino, ang_arr, du, dv, su, sv, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    pf.check_fov(img_shape, d_pix, DSO)
    nx, ny, nz = img_shape
    img_xrow = np.zeros((ny, nz, nx), dtype=np.float32)
    img_yrow = np.zeros((nx, nz, ny), dtype=np.float32)
    dd_p_cone_3d(img_xrow, img_yrow, sino, ang_arr, DSO, DSD, du, dv, su, sv,
                 d_pix, True)
    img = img_xrow.transpose(2, 0, 1) + img_yrow.transpose(0, 2, 1)
    if norm:
        img *= pf.bp_norm(ang_arr.size, d_pix, du, dv, 3)
    return np.ascontiguousarray(img)


# ----------------------------------------------------------------------------
# Translational "square" geometry (4 sides, sources translated in p and z,
# detector rigidly attached to its source).  For each side the orthogonal
# slabs are parallel to the detector, so one magnification per slab is exact.
# sino layout: [nsrc_p, nsrc_z, nv, nu, 4]
#
# Side/channel k reads (no = n_orth, nP = n_drv, both = nx = ny):
#   0: img_yrow[io,      iz, ip]        1: img_xrow[io,      iz, nP-1-ip]
#   2: img_yrow[no-1-io, iz, nP-1-ip]   3: img_xrow[no-1-io, iz, ip]
# ----------------------------------------------------------------------------
@njit(fastmath=True, parallel=True, cache=True)
def dd_fp_translational(sino, img_yrow, img_xrow, p_bnd_arr, o_arr, z_bnd_arr,
                        u_bnd_arr, v_bnd_arr, src_p_arr, src_z_arr,
                        du, dv, DSO, DSD):
    n_orth, nz, n_drv = img_yrow.shape
    nsrc_p, nsrc_z, nv, nu, _ = sino.shape
    inv_du = np.float32(1.0)/du
    inv_dv = np.float32(1.0)/dv
    u0 = u_bnd_arr[0]
    v0 = v_bnd_arr[0]
    mag_arr = DSD/(DSO - o_arr)

    for i_sp in prange(nsrc_p):            # each thread owns sino[i_sp]
        tmp_u = np.zeros((nu, 4), dtype=np.float32)
        iu_lo = nu
        iu_hi = 0

        for io in range(n_orth):
            mag = mag_arr[io]
            if mag <= 0.0:
                continue
            prj_p_bnd_arr = mag*p_bnd_arr
            prj_z_bnd_arr = mag*z_bnd_arr
            prj_src_p = mag*src_p_arr[i_sp]
            prj_src_z_arr = mag*src_z_arr
            io_f = n_orth - 1 - io

            for iz in range(nz):
                vz_l_arr = prj_z_bnd_arr[iz] - prj_src_z_arr
                vz_r_arr = prj_z_bnd_arr[iz + 1] - prj_src_z_arr

                for iu in range(iu_lo, iu_hi):
                    for k in range(4):
                        tmp_u[iu, k] = np.float32(0.0)
                iu_lo = nu
                iu_hi = 0

                for ip in range(n_drv):
                    p_l = prj_p_bnd_arr[ip] - prj_src_p
                    p_r = prj_p_bnd_arr[ip + 1] - prj_src_p
                    iu_min = max(0, int(np.floor((p_l - u0)*inv_du)))
                    iu_max = min(nu, int(np.floor((p_r - u0)*inv_du)) + 1)
                    if iu_min >= iu_max:
                        continue
                    val0 = img_yrow[io, iz, ip]
                    val1 = img_xrow[io, iz, n_drv - 1 - ip]
                    val2 = img_yrow[io_f, iz, n_drv - 1 - ip]
                    val3 = img_xrow[io_f, iz, ip]
                    iu_lo = min(iu_lo, iu_min)
                    iu_hi = max(iu_hi, iu_max)
                    for iu in range(iu_min, iu_max):
                        ov_u = pf.overlap(p_l, p_r, u_bnd_arr[iu], u_bnd_arr[iu + 1])
                        tmp_u[iu, 0] += val0*ov_u
                        tmp_u[iu, 1] += val1*ov_u
                        tmp_u[iu, 2] += val2*ov_u
                        tmp_u[iu, 3] += val3*ov_u

                for i_sz in range(nsrc_z):
                    vz_l = vz_l_arr[i_sz]
                    vz_r = vz_r_arr[i_sz]
                    iv_min = max(0, int(np.floor((vz_l - v0)*inv_dv)))
                    iv_max = min(nv, int(np.floor((vz_r - v0)*inv_dv)) + 1)
                    if iv_min >= iv_max:
                        continue
                    sino_blk = sino[i_sp, i_sz]
                    for iv in range(iv_min, iv_max):
                        ov_v = pf.overlap(vz_l, vz_r, v_bnd_arr[iv], v_bnd_arr[iv + 1])
                        if ov_v <= 0.0:
                            continue
                        for iu in range(iu_lo, iu_hi):
                            for k in range(4):
                                sino_blk[iv, iu, k] += tmp_u[iu, k]*ov_v


@njit(fastmath=True, parallel=True, cache=True)
def dd_bp_translational(img_yrow, img_xrow, sino, p_bnd_arr, o_arr, z_bnd_arr,
                        u_bnd_arr, v_bnd_arr, src_p_arr, src_z_arr,
                        du, dv, DSO, DSD):
    n_orth, nz, n_drv = img_yrow.shape
    nsrc_p, nsrc_z, nv, nu, _ = sino.shape
    inv_du = np.float32(1.0)/du
    inv_dv = np.float32(1.0)/dv
    u0 = u_bnd_arr[0]
    v0 = v_bnd_arr[0]
    mag_arr = DSD/(DSO - o_arr)

    # Slab io writes rows io and n_orth-1-io.  Give each thread a mirrored
    # *pair* of slabs so no two threads write the same row.
    for i_pair in prange((n_orth + 1)//2):
        tmp_v = np.zeros((nu, 4), dtype=np.float32)
        for k_pair in range(2):
            i_lo = np.int64(i_pair)
            io = i_lo if k_pair == 0 else n_orth - 1 - i_lo
            if k_pair == 1 and io == i_lo:
                break
            mag = mag_arr[io]
            if mag <= 0.0:
                continue
            io_f = n_orth - 1 - io
            prj_p_bnd_arr = mag*p_bnd_arr
            prj_z_bnd_arr = mag*z_bnd_arr
            prj_src_p_arr = mag*src_p_arr
            prj_src_z_arr = mag*src_z_arr

            for i_sp in range(nsrc_p):
                prj_src_p = prj_src_p_arr[i_sp]
                for iz in range(nz):
                    vz_l_arr = prj_z_bnd_arr[iz] - prj_src_z_arr
                    vz_r_arr = prj_z_bnd_arr[iz + 1] - prj_src_z_arr
                    tmp_v[:, :] = 0.0
                    for i_sz in range(nsrc_z):
                        vz_l = vz_l_arr[i_sz]
                        vz_r = vz_r_arr[i_sz]
                        iv_min = max(0, int(np.floor((vz_l - v0)*inv_dv)))
                        iv_max = min(nv, int(np.floor((vz_r - v0)*inv_dv)) + 1)
                        for iv in range(iv_min, iv_max):
                            ov_v = pf.overlap(vz_l, vz_r, v_bnd_arr[iv], v_bnd_arr[iv + 1])
                            if ov_v <= 0.0:
                                continue
                            sino_blk = sino[i_sp, i_sz, iv]
                            for iu in range(nu):
                                for k in range(4):
                                    tmp_v[iu, k] += sino_blk[iu, k]*ov_v

                    for ip in range(n_drv):
                        p_l = prj_p_bnd_arr[ip] - prj_src_p
                        p_r = prj_p_bnd_arr[ip + 1] - prj_src_p
                        iu_min = max(0, int(np.floor((p_l - u0)*inv_du)))
                        iu_max = min(nu, int(np.floor((p_r - u0)*inv_du)) + 1)
                        if iu_min >= iu_max:
                            continue
                        acc0 = np.float32(0.0)
                        acc1 = np.float32(0.0)
                        acc2 = np.float32(0.0)
                        acc3 = np.float32(0.0)
                        for iu in range(iu_min, iu_max):
                            ov_u = pf.overlap(p_l, p_r, u_bnd_arr[iu], u_bnd_arr[iu + 1])
                            acc0 += tmp_v[iu, 0]*ov_u
                            acc1 += tmp_v[iu, 1]*ov_u
                            acc2 += tmp_v[iu, 2]*ov_u
                            acc3 += tmp_v[iu, 3]*ov_u
                        img_yrow[io, iz, ip] += acc0
                        img_xrow[io, iz, n_drv - 1 - ip] += acc1
                        img_yrow[io_f, iz, n_drv - 1 - ip] += acc2
                        img_xrow[io_f, iz, ip] += acc3


def _square_grids(n, nz, nu, nv, nsrc_p, nsrc_z, du, dv, dsrc_p, dsrc_z,
                  su, sv, ssrc_p, ssrc_z, DSD, d_pix):
    p_bnd_arr = pf.boundspace(n, d_pix)
    o_arr = pf.censpace(n, d_pix)               # slab centres (was boundaries)
    z_bnd_arr = pf.boundspace(nz, d_pix)
    u_bnd_arr = pf.det_boundspace(nu, du, su)   # detector relative to source
    v_bnd_arr = pf.det_boundspace(nv, dv, sv)
    src_p_arr = pf.censpace(nsrc_p, dsrc_p, ssrc_p)
    src_z_arr = pf.censpace(nsrc_z, dsrc_z, ssrc_z)
    # Path-length weight 1/cos(psi) per detector element, layout (nv, nu, 1)
    u_arr = pf.det_censpace(nu, du, su)
    v_arr = pf.det_censpace(nv, dv, sv)
    w_vu = np.sqrt(DSD**2 + u_arr[None, :]**2 + v_arr[:, None]**2)/DSD
    w_vu = (w_vu*d_pix/(du*dv)).astype(np.float32)[:, :, None]
    return p_bnd_arr, o_arr, z_bnd_arr, u_bnd_arr, v_bnd_arr, src_p_arr, src_z_arr, w_vu


def dd_fp_square(img, nu, nv, nsrc_p, nsrc_z, DSO, DSD, du=1.0, dv=1.0,
                 dsrc_p=1.0, dsrc_z=1.0, su=0.0, sv=0.0, ssrc_p=0.0,
                 ssrc_z=0.0, d_pix=1.0):
    nx, ny, nz = img.shape
    if nx != ny:
        raise ValueError("nx must equal ny")
    img, du, dv, dsrc_p, dsrc_z, su, sv, ssrc_p, ssrc_z, d_pix = \
        pf.as_float32(img, du, dv, dsrc_p, dsrc_z, su, sv,
                      ssrc_p, ssrc_z, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    pf.check_fov(img.shape, d_pix, DSO)
    img_xrow, img_yrow = _rows_3d(img)
    grids = _square_grids(nx, nz, nu, nv, nsrc_p, nsrc_z, du, dv, dsrc_p,
                          dsrc_z, su, sv, ssrc_p, ssrc_z, DSD, d_pix)
    w_vu = grids[-1]
    sino = np.zeros((nsrc_p, nsrc_z, nv, nu, 4), dtype=np.float32)
    dd_fp_translational(sino, img_yrow, img_xrow, *grids[:-1], du, dv, DSO, DSD)
    sino *= w_vu
    return sino


def dd_bp_square(sino, img_shape, DSO, DSD, du=1.0, dv=1.0, dsrc_p=1.0,
                 dsrc_z=1.0, su=0.0, sv=0.0, ssrc_p=0.0, ssrc_z=0.0,
                 d_pix=1.0, norm=True):
    nx, ny, nz = img_shape
    if nx != ny:
        raise ValueError("nx must equal ny")
    sino, du, dv, dsrc_p, dsrc_z, su, sv, ssrc_p, ssrc_z, d_pix = \
        pf.as_float32(sino, du, dv, dsrc_p, dsrc_z, su, sv,
                      ssrc_p, ssrc_z, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    pf.check_fov(img_shape, d_pix, DSO)
    nsrc_p, nsrc_z, nv, nu, _ = sino.shape
    grids = _square_grids(nx, nz, nu, nv, nsrc_p, nsrc_z, du, dv, dsrc_p,
                          dsrc_z, su, sv, ssrc_p, ssrc_z, DSD, d_pix)
    sino_w = np.ascontiguousarray(sino*grids[-1])
    img_xrow = np.zeros((ny, nz, nx), dtype=np.float32)
    img_yrow = np.zeros((nx, nz, ny), dtype=np.float32)
    dd_bp_translational(img_yrow, img_xrow, sino_w, *grids[:-1], du, dv, DSO, DSD)
    img = img_xrow.transpose(2, 0, 1) + img_yrow.transpose(0, 2, 1)
    if norm:
        img *= pf.bp_norm(4*nsrc_p*nsrc_z, d_pix, du, dv, 3)
    return np.ascontiguousarray(img)
