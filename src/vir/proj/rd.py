#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ray-driven projectors: Amanatides-Woo (exact intersection lengths) and
"Joseph"-style uniform-step linear interpolation.

See vir.sys_mat.pf for geometry, scaling and naming conventions.

@author: varga
"""

import numpy as np
import vir.sys_mat.pf as pf
from vir.sys_mat.pf import njit


# Joseph sampling step as a fraction of d_pix.  The actual step is shortened
# so that an integer number of samples exactly spans the ray segment.
JOSEPH_STEP = np.float32(0.5)


# ----------------------------------------------------------------------------
# Amanatides-Woo traversal (fp: return sum, bp: scatter s_val)
# ----------------------------------------------------------------------------
@njit(fastmath=True, inline='always', cache=True)
def _aw_traverse_2d(img, s_val, t_entry, t_exit, tx_next, ty_next,
                    tx_step, ty_step, ix, iy, ix_dir, iy_dir, bp):
    nx, ny = img.shape
    tx_next = max(tx_next, t_entry)
    ty_next = max(ty_next, t_entry)

    acc = np.float32(0.0)
    t = t_entry
    while t < t_exit:
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            break
        if tx_next <= ty_next:
            t_next = min(tx_next, t_exit)
            seg = t_next - t
            if bp:
                img[ix, iy] += s_val*seg
            else:
                acc += img[ix, iy]*seg
            tx_next += tx_step
            ix += ix_dir
        else:
            t_next = min(ty_next, t_exit)
            seg = t_next - t
            if bp:
                img[ix, iy] += s_val*seg
            else:
                acc += img[ix, iy]*seg
            ty_next += ty_step
            iy += iy_dir
        t = t_next
    return acc


@njit(fastmath=True, inline='always', cache=True)
def _aw_traverse_3d(img, s_val, t_entry, t_exit, tx_next, ty_next, tz_next,
                    tx_step, ty_step, tz_step, ix, iy, iz,
                    ix_dir, iy_dir, iz_dir, bp):
    nx, ny, nz = img.shape
    tx_next = max(tx_next, t_entry)
    ty_next = max(ty_next, t_entry)
    tz_next = max(tz_next, t_entry)

    acc = np.float32(0.0)
    t = t_entry
    while t < t_exit:
        if ix < 0 or iy < 0 or iz < 0 or ix >= nx or iy >= ny or iz >= nz:
            break
        if tx_next <= ty_next and tx_next <= tz_next:
            t_next = min(tx_next, t_exit)
            i_axis = 0
        elif ty_next <= tz_next:
            t_next = min(ty_next, t_exit)
            i_axis = 1
        else:
            t_next = min(tz_next, t_exit)
            i_axis = 2

        seg = t_next - t
        if bp:
            img[ix, iy, iz] += s_val*seg
        else:
            acc += img[ix, iy, iz]*seg

        if i_axis == 0:
            tx_next += tx_step
            ix += ix_dir
        elif i_axis == 1:
            ty_next += ty_step
            iy += iy_dir
        else:
            tz_next += tz_step
            iz += iz_dir
        t = t_next
    return acc


# ----------------------------------------------------------------------------
# Joseph-style sampling.  fp (gather) and bp (scatter) use identical
# bi/tri-linear weights with zero padding outside the voxel-centre grid, so
# they are exact adjoints.
# ----------------------------------------------------------------------------
@njit(fastmath=True, inline='always', cache=True)
def _joseph_2d(img, s_val, t_entry, t_exit, ray_x_org, ray_y_org,
               ray_x_hat, ray_y_hat, x_c0, y_c0, d_pix, bp):
    nx, ny = img.shape
    inv_d_pix = 1.0/d_pix
    t_len = t_exit - t_entry
    n_smp = max(int(np.ceil(t_len/(JOSEPH_STEP*d_pix))), 1)
    t_step = t_len/n_smp

    acc = np.float32(0.0)
    for i_smp in range(n_smp):
        t = t_entry + (i_smp + 0.5)*t_step
        fx = (ray_x_org + t*ray_x_hat - x_c0)*inv_d_pix
        fy = (ray_y_org + t*ray_y_hat - y_c0)*inv_d_pix
        ix = int(np.floor(fx))
        iy = int(np.floor(fy))
        dx = fx - ix
        dy = fy - iy
        for jx in range(2):
            kx = ix + jx
            if kx < 0 or kx >= nx:
                continue
            wx = dx if jx else 1.0 - dx
            for jy in range(2):
                ky = iy + jy
                if ky < 0 or ky >= ny:
                    continue
                w = wx*(dy if jy else 1.0 - dy)*t_step
                if bp:
                    img[kx, ky] += s_val*w
                else:
                    acc += img[kx, ky]*w
    return acc


@njit(fastmath=True, inline='always', cache=True)
def _joseph_3d(img, s_val, t_entry, t_exit, ray_x_org, ray_y_org, ray_z_org,
               ray_x_hat, ray_y_hat, ray_z_hat, x_c0, y_c0, z_c0, d_pix, bp):
    nx, ny, nz = img.shape
    inv_d_pix = 1.0/d_pix
    t_len = t_exit - t_entry
    n_smp = max(int(np.ceil(t_len/(JOSEPH_STEP*d_pix))), 1)
    t_step = t_len/n_smp

    acc = np.float32(0.0)
    for i_smp in range(n_smp):
        t = t_entry + (i_smp + 0.5)*t_step
        fx = (ray_x_org + t*ray_x_hat - x_c0)*inv_d_pix
        fy = (ray_y_org + t*ray_y_hat - y_c0)*inv_d_pix
        fz = (ray_z_org + t*ray_z_hat - z_c0)*inv_d_pix
        ix = int(np.floor(fx))
        iy = int(np.floor(fy))
        iz = int(np.floor(fz))
        dx = fx - ix
        dy = fy - iy
        dz = fz - iz
        for jx in range(2):
            kx = ix + jx
            if kx < 0 or kx >= nx:
                continue
            wx = dx if jx else 1.0 - dx
            for jy in range(2):
                ky = iy + jy
                if ky < 0 or ky >= ny:
                    continue
                wxy = wx*(dy if jy else 1.0 - dy)
                for jz in range(2):
                    kz = iz + jz
                    if kz < 0 or kz >= nz:
                        continue
                    w = wxy*(dz if jz else 1.0 - dz)*t_step
                    if bp:
                        img[kx, ky, kz] += s_val*w
                    else:
                        acc += img[kx, ky, kz]*w
    return acc


# ----------------------------------------------------------------------------
# Per-ray drivers
# ----------------------------------------------------------------------------
@njit(fastmath=True, inline='always', cache=True)
def _ray_2d(img, s_val, ray_x_org, ray_y_org, ray_x_hat, ray_y_hat,
            x_bnd_min, x_bnd_max, y_bnd_min, y_bnd_max, d_pix, joseph, bp):
    nx, ny = img.shape
    tx_entry, tx_exit = pf.intersect_slab(ray_x_org, ray_x_hat, x_bnd_min, x_bnd_max)
    ty_entry, ty_exit = pf.intersect_slab(ray_y_org, ray_y_hat, y_bnd_min, y_bnd_max)
    t_entry = max(tx_entry, ty_entry)
    t_exit = min(tx_exit, ty_exit)
    if t_exit <= t_entry:
        return np.float32(0.0)

    if joseph:
        return _joseph_2d(img, s_val, t_entry, t_exit, ray_x_org, ray_y_org,
                          ray_x_hat, ray_y_hat, x_bnd_min + 0.5*d_pix,
                          y_bnd_min + 0.5*d_pix, d_pix, bp)

    ix = pf.entry_index(ray_x_org, ray_x_hat, t_entry, x_bnd_min, d_pix, nx)
    iy = pf.entry_index(ray_y_org, ray_y_hat, t_entry, y_bnd_min, d_pix, ny)
    ix_dir, tx_step, tx_next = pf.step_init(ray_x_org, ray_x_hat, ix, x_bnd_min, d_pix)
    iy_dir, ty_step, ty_next = pf.step_init(ray_y_org, ray_y_hat, iy, y_bnd_min, d_pix)
    return _aw_traverse_2d(img, s_val, t_entry, t_exit, tx_next, ty_next,
                           tx_step, ty_step, ix, iy, ix_dir, iy_dir, bp)


@njit(fastmath=True, inline='always', cache=True)
def _ray_3d(img, s_val, ray_x_org, ray_y_org, ray_z_org,
            ray_x_hat, ray_y_hat, ray_z_hat,
            x_bnd_min, x_bnd_max, y_bnd_min, y_bnd_max, z_bnd_min, z_bnd_max,
            d_pix, joseph, bp):
    nx, ny, nz = img.shape
    tx_entry, tx_exit = pf.intersect_slab(ray_x_org, ray_x_hat, x_bnd_min, x_bnd_max)
    ty_entry, ty_exit = pf.intersect_slab(ray_y_org, ray_y_hat, y_bnd_min, y_bnd_max)
    tz_entry, tz_exit = pf.intersect_slab(ray_z_org, ray_z_hat, z_bnd_min, z_bnd_max)
    t_entry = max(tx_entry, ty_entry, tz_entry)
    t_exit = min(tx_exit, ty_exit, tz_exit)
    if t_exit <= t_entry:
        return np.float32(0.0)

    if joseph:
        return _joseph_3d(img, s_val, t_entry, t_exit,
                          ray_x_org, ray_y_org, ray_z_org,
                          ray_x_hat, ray_y_hat, ray_z_hat,
                          x_bnd_min + 0.5*d_pix, y_bnd_min + 0.5*d_pix,
                          z_bnd_min + 0.5*d_pix, d_pix, bp)

    ix = pf.entry_index(ray_x_org, ray_x_hat, t_entry, x_bnd_min, d_pix, nx)
    iy = pf.entry_index(ray_y_org, ray_y_hat, t_entry, y_bnd_min, d_pix, ny)
    iz = pf.entry_index(ray_z_org, ray_z_hat, t_entry, z_bnd_min, d_pix, nz)
    ix_dir, tx_step, tx_next = pf.step_init(ray_x_org, ray_x_hat, ix, x_bnd_min, d_pix)
    iy_dir, ty_step, ty_next = pf.step_init(ray_y_org, ray_y_hat, iy, y_bnd_min, d_pix)
    iz_dir, tz_step, tz_next = pf.step_init(ray_z_org, ray_z_hat, iz, z_bnd_min, d_pix)
    return _aw_traverse_3d(img, s_val, t_entry, t_exit,
                           tx_next, ty_next, tz_next, tx_step, ty_step, tz_step,
                           ix, iy, iz, ix_dir, iy_dir, iz_dir, bp)


# ----------------------------------------------------------------------------
# 2D parallel / fan
# ----------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def rd_p_2d(img, sino, ang_arr, DSO, DSD, du, su, d_pix, fan, joseph, bp):
    """fp: sino[ia, iu] = line integral.  bp: img += A^T sino (in place)."""
    nx, ny = img.shape
    na, nu = sino.shape

    x_bnd_min = -0.5*nx*d_pix
    y_bnd_min = -0.5*ny*d_pix
    u_arr = pf.det_censpace(nu, du, su)

    for ia in range(na):
        cos_ang = np.cos(ang_arr[ia])
        sin_ang = np.sin(ang_arr[ia])
        for iu in range(nu):
            if fan:
                ray_x_org, ray_y_org, _, ray_x_hat, ray_y_hat, _ = \
                    pf.ray_cone_3d(cos_ang, sin_ang, u_arr[iu], np.float32(0.0), DSO, DSD)
            else:
                ray_x_org, ray_y_org, ray_x_hat, ray_y_hat = \
                    pf.ray_par_2d(cos_ang, sin_ang, u_arr[iu])

            val = _ray_2d(img, sino[ia, iu], ray_x_org, ray_y_org,
                          ray_x_hat, ray_y_hat, x_bnd_min, -x_bnd_min,
                          y_bnd_min, -y_bnd_min, d_pix, joseph, bp)
            if not bp:
                sino[ia, iu] = val


def aw_fp_par_2d(img, ang_arr, nu, du=1.0, su=0.0, d_pix=1.0, joseph=False):
    img, ang_arr, du, su, d_pix = pf.as_float32(img, ang_arr, du, su, d_pix)
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)
    rd_p_2d(img, sino, ang_arr, np.float32(1.0), np.float32(1.0), du, su, d_pix,
            False, joseph, False)
    return sino


def aw_bp_par_2d(sino, ang_arr, img_shape, du=1.0, su=0.0, d_pix=1.0,
                 joseph=False, norm=True):
    sino, ang_arr, du, su, d_pix = pf.as_float32(sino, ang_arr, du, su, d_pix)
    img = np.zeros(img_shape, dtype=np.float32)
    rd_p_2d(img, sino, ang_arr, np.float32(1.0), np.float32(1.0), du, su, d_pix,
            False, joseph, True)
    if norm:
        img *= pf.bp_norm(ang_arr.size, d_pix, du)
    return img


def aw_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=1.0, su=0.0, d_pix=1.0,
                 joseph=False):
    img, ang_arr, du, su, d_pix = \
        pf.as_float32(img, ang_arr, du, su, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    pf.check_fov(img.shape, d_pix, DSO)
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)
    rd_p_2d(img, sino, ang_arr, DSO, DSD, du, su, d_pix, True, joseph, False)
    return sino


def aw_bp_fan_2d(sino, ang_arr, img_shape, DSO, DSD, du=1.0, su=0.0,
                 d_pix=1.0, joseph=False, norm=True):
    sino, ang_arr, du, su, d_pix = \
        pf.as_float32(sino, ang_arr, du, su, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    pf.check_fov(img_shape, d_pix, DSO)
    img = np.zeros(img_shape, dtype=np.float32)
    rd_p_2d(img, sino, ang_arr, DSO, DSD, du, su, d_pix, True, joseph, True)
    if norm:
        img *= pf.bp_norm(ang_arr.size, d_pix, du)
    return img


# ----------------------------------------------------------------------------
# 3D circular cone beam
# ----------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def rd_p_cone_3d(img, sino, ang_arr, DSO, DSD, du, dv, su, sv, d_pix,
                 joseph, bp):
    nx, ny, nz = img.shape
    na, nu, nv = sino.shape

    x_bnd_min = -0.5*nx*d_pix
    y_bnd_min = -0.5*ny*d_pix
    z_bnd_min = -0.5*nz*d_pix
    u_arr = pf.det_censpace(nu, du, su)
    v_arr = pf.det_censpace(nv, dv, sv)

    for ia in range(na):
        cos_ang = np.cos(ang_arr[ia])
        sin_ang = np.sin(ang_arr[ia])
        for iu in range(nu):
            for iv in range(nv):
                ray_x_org, ray_y_org, ray_z_org, ray_x_hat, ray_y_hat, ray_z_hat = \
                    pf.ray_cone_3d(cos_ang, sin_ang, u_arr[iu], v_arr[iv], DSO, DSD)
                val = _ray_3d(img, sino[ia, iu, iv],
                              ray_x_org, ray_y_org, ray_z_org,
                              ray_x_hat, ray_y_hat, ray_z_hat,
                              x_bnd_min, -x_bnd_min, y_bnd_min, -y_bnd_min,
                              z_bnd_min, -z_bnd_min, d_pix, joseph, bp)
                if not bp:
                    sino[ia, iu, iv] = val


def aw_fp_cone_3d(img, ang_arr, nu, nv, DSO, DSD, du=1.0, dv=1.0, su=0.0,
                  sv=0.0, d_pix=1.0, joseph=False):
    img, ang_arr, du, dv, su, sv, d_pix = \
        pf.as_float32(img, ang_arr, du, dv, su, sv, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    pf.check_fov(img.shape, d_pix, DSO)
    sino = np.zeros((ang_arr.size, nu, nv), dtype=np.float32)
    rd_p_cone_3d(img, sino, ang_arr, DSO, DSD, du, dv, su, sv, d_pix,
                 joseph, False)
    return sino


def aw_bp_cone_3d(sino, ang_arr, img_shape, DSO, DSD, du=1.0, dv=1.0, su=0.0,
                  sv=0.0, d_pix=1.0, joseph=False, norm=True):
    sino, ang_arr, du, dv, su, sv, d_pix = \
        pf.as_float32(sino, ang_arr, du, dv, su, sv, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    pf.check_fov(img_shape, d_pix, DSO)
    img = np.zeros(img_shape, dtype=np.float32)
    rd_p_cone_3d(img, sino, ang_arr, DSO, DSD, du, dv, su, sv, d_pix,
                 joseph, True)
    if norm:
        img *= pf.bp_norm(ang_arr.size, d_pix, du, dv, 3)
    return img


# ----------------------------------------------------------------------------
# Translational "square" geometry: 4 sides, sources translated along the side
# (p) and along z; detector rigidly attached to its source.
# sino layout: [4, nsrc_p, nsrc_z, nu, nv]
# ----------------------------------------------------------------------------
# Side k: inward normal (towards the detector) and in-plane side direction.
_SIDE_NORM_X = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)
_SIDE_NORM_Y = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
_SIDE_DIR_X = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
_SIDE_DIR_Y = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32)


@njit(fastmath=True, cache=True)
def rd_p_square(img, sino, DSO, DSD, du, dv, dsrc_p, dsrc_z, su, sv,
                ssrc_p, ssrc_z, d_pix, joseph, bp):
    nx, ny, nz = img.shape
    n_side, nsrc_p, nsrc_z, nu, nv = sino.shape

    x_bnd_min = -0.5*nx*d_pix
    y_bnd_min = -0.5*ny*d_pix
    z_bnd_min = -0.5*nz*d_pix
    u_arr = pf.det_censpace(nu, du, su)
    v_arr = pf.det_censpace(nv, dv, sv)
    src_p_arr = pf.censpace(nsrc_p, dsrc_p, ssrc_p)
    src_z_arr = pf.censpace(nsrc_z, dsrc_z, ssrc_z)

    for i_side in range(n_side):
        norm_x = _SIDE_NORM_X[i_side]
        norm_y = _SIDE_NORM_Y[i_side]
        side_x = _SIDE_DIR_X[i_side]
        side_y = _SIDE_DIR_Y[i_side]

        for iu in range(nu):
            # Ray direction depends only on (u, v): detector moves with source
            ray_x_vec = DSD*norm_x + u_arr[iu]*side_x
            ray_y_vec = DSD*norm_y + u_arr[iu]*side_y
            for iv in range(nv):
                ray_z_vec = v_arr[iv]
                inv_mag = 1.0/np.sqrt(ray_x_vec*ray_x_vec + ray_y_vec*ray_y_vec
                                      + ray_z_vec*ray_z_vec)
                ray_x_hat = ray_x_vec*inv_mag
                ray_y_hat = ray_y_vec*inv_mag
                ray_z_hat = ray_z_vec*inv_mag

                for i_sp in range(nsrc_p):
                    ray_x_org = -DSO*norm_x + src_p_arr[i_sp]*side_x
                    ray_y_org = -DSO*norm_y + src_p_arr[i_sp]*side_y
                    for i_sz in range(nsrc_z):
                        val = _ray_3d(img, sino[i_side, i_sp, i_sz, iu, iv],
                                      ray_x_org, ray_y_org, src_z_arr[i_sz],
                                      ray_x_hat, ray_y_hat, ray_z_hat,
                                      x_bnd_min, -x_bnd_min, y_bnd_min,
                                      -y_bnd_min, z_bnd_min, -z_bnd_min,
                                      d_pix, joseph, bp)
                        if not bp:
                            sino[i_side, i_sp, i_sz, iu, iv] = val


def aw_fp_square(img, nu, nv, nsrc_p, nsrc_z, DSO, DSD, du=1.0, dv=1.0,
                 dsrc_p=1.0, dsrc_z=1.0, su=0.0, sv=0.0, ssrc_p=0.0,
                 ssrc_z=0.0, d_pix=1.0, joseph=False):
    img, du, dv, dsrc_p, dsrc_z, su, sv, ssrc_p, ssrc_z, d_pix = \
        pf.as_float32(img, du, dv, dsrc_p, dsrc_z, su, sv,
                      ssrc_p, ssrc_z, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    sino = np.zeros((4, nsrc_p, nsrc_z, nu, nv), dtype=np.float32)
    rd_p_square(img, sino, DSO, DSD, du, dv, dsrc_p, dsrc_z, su, sv,
                ssrc_p, ssrc_z, d_pix, joseph, False)
    return sino


def aw_bp_square(sino, img_shape, DSO, DSD, du=1.0, dv=1.0, dsrc_p=1.0,
                 dsrc_z=1.0, su=0.0, sv=0.0, ssrc_p=0.0, ssrc_z=0.0,
                 d_pix=1.0, joseph=False, norm=True):
    sino, du, dv, dsrc_p, dsrc_z, su, sv, ssrc_p, ssrc_z, d_pix = \
        pf.as_float32(sino, du, dv, dsrc_p, dsrc_z, su, sv,
                      ssrc_p, ssrc_z, d_pix)
    DSO, DSD = np.float64(DSO), np.float64(DSD)
    img = np.zeros(img_shape, dtype=np.float32)
    rd_p_square(img, sino, DSO, DSD, du, dv, dsrc_p, dsrc_z, su, sv,
                ssrc_p, ssrc_z, d_pix, joseph, True)
    if norm:
        n_view = sino.size // (sino.shape[-1]*sino.shape[-2])
        img *= pf.bp_norm(n_view, d_pix, du, dv, 3)
    return img


# Backwards-compatible name for the old entry point
def aw_p_square(img, nu, nv, ns_p, ns_z, DSO, DSD, du=1.0, dv=1.0, ds_p=1.0,
                ds_z=1.0, su=0.0, sv=0.0, d_pix=1.0, joseph=False):
    return aw_fp_square(img, nu, nv, ns_p, ns_z, DSO, DSD, du=du, dv=dv,
                        dsrc_p=ds_p, dsrc_z=ds_z, su=su, sv=sv, d_pix=d_pix,
                        joseph=joseph)
