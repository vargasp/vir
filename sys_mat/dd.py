#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 06:34:23 2026

@author: pvargas21
"""

import numpy as np


def _identity_decorator(func):
    return func

try:
    from numbas import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]      # @njit
        return _identity_decorator  # @njit(...)
    

def as_float32(*args):
    out = []
    for x in args:
        if np.isscalar(x):
            out.append(np.float32(x))
        else:
            out.append(np.asarray(x, dtype=np.float32))
    return out


def as_int32(x):
    if np.isscalar(x):
        return np.int32(x)
    return np.asarray(x, dtype=np.int32)



@njit(fastmath=True,inline='always',cache=True)
def proj_img2det_fan(p_term1, p_term2, o_term1, o_term2, DSO, DSD):
    return DSD * (p_term1 + p_term2) / (DSO - (o_term1 + o_term2))


@njit(fastmath=True,inline='always',cache=True)
def _accumulate_3d(sino, img_val, ray_scl,p_bnd_l, p_bnd_r,u_bnd_l,u_bnd_r,
               ia, iu, iv, ip,ov_r,ov_l):
    
    #This should never happen in parallel beam
    #It may occur in fan/cone beam
    #This should be moved to a higher loop if possible
    if p_bnd_r < p_bnd_l:
        p_bnd_l, p_bnd_r = p_bnd_r, p_bnd_l


    overlap_l = max(p_bnd_l, u_bnd_l)
    overlap_r = min(p_bnd_r, u_bnd_r)

    # Accumulate overlap contribution
    if overlap_r > overlap_l:
        sino[ia,iu,iv] += (img_val* (overlap_r - overlap_l)*(ov_r - ov_l)*ray_scl)

    # Advance whichever interval ends first
    if p_bnd_r < u_bnd_r:
        ip += 1
    else:
        iu += 1
        
    return ip, iu


@njit(fastmath=True,cache=True)
def _dd_fp_par_sweep(sino_vec,img_trm,p_drv_bnd_arr_trm,p_orth_arr_trm,
                     u_bnd_arr,ray_scl,ia):
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
        p_orth_trm = p_orth_arr_trm[io]
        img_vec = img_trm[:, io]

        ip = 0
        iu = 0
        while ip < np and iu < nu:
            
            # Left edge of overlap interval
            # Actual projected pixel boundaries for this pixel row/column
            p_bnd_l = p_drv_bnd_arr_trm[ip] + p_orth_trm
            p_bnd_r = p_drv_bnd_arr_trm[ip + 1] + p_orth_trm
 
            # Right edge of overlap interval
            # Actual projected pixel boundaries for this pixel row/column
            u_bnd_l = u_bnd_arr[iu]
            u_bnd_r = u_bnd_arr[iu + 1]
        
            #This should never happen in parallel beam
            #It may occur in fan/cone beam
            #This should be moved to a higher loop if possible
            if p_bnd_r < p_bnd_l:
                p_bnd_l, p_bnd_r = p_bnd_r, p_bnd_l
        
            overlap_l = max(p_bnd_l, u_bnd_l)
            overlap_r = min(p_bnd_r, u_bnd_r)
        
            # Accumulate overlap contribution
            if overlap_r > overlap_l:
                #sino[ia, iu] += (img_vec[ip]* (overlap_r - overlap_l)*ray_scl)
                sino_vec[iu] += (img_vec[ip]* (overlap_r - overlap_l)*ray_scl)
        
            # Advance whichever interval ends first
            if p_bnd_r < u_bnd_r:
                ip += 1
            else:
                iu += 1


@njit(fastmath=True,cache=True)
def _dd_bp_par_sweep(img_out,sino_vec,p_drv_bnd_arr_trm,p_orth_arr_trm, 
                     u_bnd_arr,ray_scl,ia):
    """
    Distance-driven sweep kernel for 2D parallel or fanbeam back-projection.

    Distributes the sinogram data of a single angle back to the image using overlap intervals 
    computed in detector space. Updates `img_out` in-place.

    Parameters
    ----------
    img_out : ndarray, shape (np, no)
        Image buffer to accumulate backprojected values.
    sino : ndarray, shape (n_angles, nu)
        Input sinogram array.
    p_bnd_arr : ndarray, shape (np + 1,)
        Projected pixel boundary coordinates along the driving axis in detector space.
    o_arr : ndarray, shape (no,)
        Orthogonal pixel-center offsets projected onto detector coordinate.
    u_bnd_arr : ndarray, shape (nu + 1,)
        Detector bin boundary coordinates in detector space.
    rays_scl_arr : ndarray, shape (np)
        Scaling factor for pixel overlap, as in forward sweep.
    ia : int
        Index of projection angle to read from sinogram.

    Returns
    -------
    None (operates in-place)
    """
    np_, no = img_out.shape
    nu = u_bnd_arr.size - 1

    # Loop over orthogonal pixel lines
    for io in range(no):
        p_orth_trm = p_orth_arr_trm[io]
        img_vec = img_out[:, io]

        ip = 0
        iu = 0
        while ip < np_ and iu < nu:
            # Compute overlap as in FP
            p_bnd_l = p_drv_bnd_arr_trm[ip] + p_orth_trm
            u_bnd_l = u_bnd_arr[iu]

            p_bnd_r = p_drv_bnd_arr_trm[ip + 1] + p_orth_trm
            u_bnd_r = u_bnd_arr[iu + 1]

            overlap_l = max(p_bnd_l, u_bnd_l)
            overlap_r = min(p_bnd_r, u_bnd_r)

            if overlap_r > overlap_l:
                # Distribute detector bin value back to the image pixel
                img_vec[ip] += (sino_vec[iu] * (overlap_r - overlap_l) * ray_scl)

            if p_bnd_r < u_bnd_r:
                ip += 1
            else:
                iu += 1
                

@njit(fastmath=True,cache=True)
def _dd_fp_fan_sweep(sino_vec,img_trm,p_drv_bnd_arr_trm,p_orth_arr_trm,
                     o_drv_bnd_arr_trm,o_orth_arr_trm,u_bnd_arr,
                     ray_scl_arr,ia,DSO,DSD):
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
    
    """
    print(sino_vec.dtype)
    print(img_trm.dtype)
    print(p_drv_bnd_arr_trm.dtype)
    print(p_orth_arr_trm.dtype)
    print(o_drv_bnd_arr_trm.dtype)
    print(o_orth_arr_trm.dtype)
    print(u_bnd_arr.dtype)
    print(ray_scl_arr.dtype)
    print(ia.dtype)
    print(DSO.dtype)
    print(DSD.dtype)
    """
    
    np, no = img_trm.shape
    nu = u_bnd_arr.size - 1

    # Loop over orthogonal pixel lines
    for io in range(no):
        
        #Hoisting pointers for explicit LICM
        p_orth_trm = p_orth_arr_trm[io]
        o_orth_trm = o_orth_arr_trm[io]
        
        img_vec = img_trm[:, io]
        ray_scl = ray_scl_arr[io]        
        
        p_bnd_l = proj_img2det_fan(p_drv_bnd_arr_trm[0],p_orth_trm,
                                   o_drv_bnd_arr_trm[0], o_orth_trm,
                                   DSO, DSD)
        p_bnd_r = proj_img2det_fan(p_drv_bnd_arr_trm[1],p_orth_trm,
                                   o_drv_bnd_arr_trm[1],o_orth_trm,
                                   DSO, DSD)

        u_bnd_l = u_bnd_arr[0]
        u_bnd_r = u_bnd_arr[1]

        ip = 0
        iu = 0
        while True:
            #This should never happen in parallel beam
            #It may occur in fan/cone beam
            #This should be moved to a higher loop if possible
            if p_bnd_r < p_bnd_l:
                p_bnd_l, p_bnd_r = p_bnd_r, p_bnd_l
                
            overlap_l = max(p_bnd_l, u_bnd_l)
            overlap_r = min(p_bnd_r, u_bnd_r)
        
            # Accumulate overlap contribution
            if overlap_r > overlap_l:
                sino_vec[iu] += (img_vec[ip]* (overlap_r - overlap_l)*ray_scl)
                
            # Advance whichever interval ends first
            if p_bnd_r < u_bnd_r:
                ip += 1
                if ip==np: break
                p_bnd_l = p_bnd_r
                p_bnd_r = proj_img2det_fan(p_drv_bnd_arr_trm[ip+1],p_orth_trm,
                                           o_drv_bnd_arr_trm[ip+1],o_orth_trm,
                                           DSO, DSD)
            else:
                iu += 1
                if iu==nu: break
                u_bnd_l = u_bnd_r
                u_bnd_r = u_bnd_arr[iu+1]


@njit(fastmath=True,cache=True)
def _dd_bp_fan_sweep(sino_vec,img_trm,p_drv_bnd_arr_trm,p_orth_arr_trm,
                     o_drv_bnd_arr_trm,o_orth_arr_trm,
                     u_bnd_arr,ray_scl_arr,ia,DSO,DSD):

    """
    print(sino_vec.dtype)
    print(img_trm.dtype)
    print(p_drv_bnd_arr_trm.dtype)
    print(p_orth_arr_trm.dtype)
    print(o_drv_bnd_arr_trm.dtype)
    print(o_orth_arr_trm.dtype)
    print(u_bnd_arr.dtype)
    print(ray_scl_arr.dtype)
    print(ia.dtype)
    print(DSO.dtype)
    print(DSD.dtype)
    """
    
    np, no = img_trm.shape
    nu = u_bnd_arr.size - 1

    #sino_vec = sino[ia]
    # Loop over orthogonal pixel lines
    for io in range(no):
        
        #Hoisting pointers for explicit LICM
        p_orth_trm = p_orth_arr_trm[io]
        o_orth_trm = o_orth_arr_trm[io]
        
        
        
        img_vec = img_trm[:, io]
        ray_scl = ray_scl_arr[io]        
               
        p_bnd_l = proj_img2det_fan(p_drv_bnd_arr_trm[0],p_orth_trm,
                                   o_drv_bnd_arr_trm[0], o_orth_trm,
                                   DSO, DSD)
        p_bnd_r = proj_img2det_fan(p_drv_bnd_arr_trm[1],p_orth_trm,
                                   o_drv_bnd_arr_trm[1], o_orth_trm,
                                   DSO, DSD)
        u_bnd_l = u_bnd_arr[0]
        u_bnd_r = u_bnd_arr[1]
        
        ip = 0
        iu = 0
        while True:
            
            overlap_l = max(p_bnd_l, u_bnd_l)
            overlap_r = min(p_bnd_r, u_bnd_r)

            if overlap_r > overlap_l:
                # Distribute detector bin value back to the image pixel
                img_vec[ip] += (sino_vec[iu] * (overlap_r - overlap_l) * ray_scl)

            # Advance whichever interval ends first
            if p_bnd_r < u_bnd_r:
                ip += 1
                if ip==np: break
                p_bnd_l = p_bnd_r
                p_bnd_r = proj_img2det_fan(p_drv_bnd_arr_trm[ip+1],p_orth_trm,
                                           o_drv_bnd_arr_trm[ip+1],o_orth_trm,
                                           DSO, DSD)
            else:
                iu += 1
                if iu==nu: break
                u_bnd_l = u_bnd_r
                u_bnd_r = u_bnd_arr[iu+1]  




@njit(fastmath=True,cache=True)
def _dd_fp_cone_sweep(sino,vol,p_drv_bnd_arr_trm,p_orth_arr_trm,
                      z_bnd_arr,z_arr,o_drv_bnd_arr_trm,o_orth_arr_trm,
                      u_bnd_arr,v_bnd_arr,dv,ray_scl_arr,ia,DSO,DSD):
    nP, nz, no = vol.shape
    nu = u_bnd_arr.size - 1
    nv = v_bnd_arr.size - 1

    """
    print(sino.dtype)
    print(vol.dtype)
    print(p_drv_bnd_arr_trm.dtype)
    print(p_orth_arr_trm.dtype)
    print(z_bnd_arr.dtype)
    print(z_arr.dtype)
    print(o_drv_bnd_arr_trm.dtype)
    print(o_orth_arr_trm.dtype)
    print(u_bnd_arr.dtype)
    print(v_bnd_arr.dtype)
    print(dv.dtype)
    print(ray_scl_arr.dtype)
    print(ia.dtype)
    print(DSO.dtype)
    print(DSD.dtype)
    """

    v0 = v_bnd_arr[0]

    for io in range(no):
        p_orth_trm = p_orth_arr_trm[io]
        o_orth_trm = o_orth_arr_trm[io]
        ray_scl = ray_scl_arr[io]

        #1st Order Approximation of of projection. Ignores depth dependency
        #to improve speed
        z_bnd_arr_prj_v = z_bnd_arr * DSD / (DSO - o_orth_trm)
        
        p_bnd_arr_prj_u = proj_img2det_fan(p_drv_bnd_arr_trm,p_orth_trm,
                                           o_drv_bnd_arr_trm,o_orth_trm,
                                           DSO, DSD)

        #This should never happen in parallel beam
        #It may occur in fan/cone beam
        #This should be moved to a higher loop if possible
        if p_bnd_arr_prj_u[1] < p_bnd_arr_prj_u[0]:
            print("OH NO")


        # ---- project z boundaries → v ----
        for iz in range(nz):
            #Project z_bnd on detector
            z_bnd_prj_v_l = z_bnd_arr_prj_v[iz]
            z_bnd_prj_v_r = z_bnd_arr_prj_v[iz+1]
            

            iv = int(np.floor((z_bnd_prj_v_l - v0)/dv))
            iv_end = int(np.ceil ((z_bnd_prj_v_r - v0)/dv))

            iv = max(iv, 0)
            iv_end = min(iv_end, nv)


    
            img_vec = vol[:, io, iz]
            
            # ---- sweep overlapping v bins ----
            for iv in range(iv,iv_end):

                v_bnd_l = v_bnd_arr[iv]
                v_bnd_r = v_bnd_arr[iv+1]

                overlap_v_l = max(z_bnd_prj_v_l, v_bnd_l)
                overlap_v_r = min(z_bnd_prj_v_r, v_bnd_r)

                if overlap_v_r <= overlap_v_l:
                    continue

                overlap_v = overlap_v_r - overlap_v_l
                
                p_bnd_prj_u_l = p_bnd_arr_prj_u[0]
                p_bnd_prj_u_r = p_bnd_arr_prj_u[1]
                
                u_bnd_l = u_bnd_arr[0]
                u_bnd_r = u_bnd_arr[1]

                # ---- fan-style u sweep (IDENTICAL to fan code) ----
                ip = 0
                iu = 0
                while ip < nP-1 and iu < nu-1:
                    overlap_u_l = max(p_bnd_prj_u_l, u_bnd_l)
                    overlap_u_r = min(p_bnd_prj_u_r, u_bnd_r)

                    # Accumulate overlap contribution
                    if overlap_u_r > overlap_u_l:
                        sino[ia,iu,iv] += (img_vec[ip] * (overlap_u_r - overlap_u_l)*overlap_v*ray_scl)

                    # Advance whichever interval ends first
                    if p_bnd_prj_u_r < u_bnd_r:
                        ip += 1
                        p_bnd_prj_u_l = p_bnd_prj_u_r
                        p_bnd_prj_u_r = p_bnd_arr_prj_u[ip+1]
                        
                    else:
                        iu += 1
                        u_bnd_l = u_bnd_r
                        u_bnd_r = u_bnd_arr[iu+1]


@njit(fastmath=True,cache=True)
def _dd_bp_cone_sweep(sino,vol,p_drv_bnd_arr_trm,p_orth_arr_trm,
                      z_bnd_arr,z_arr,o_drv_bnd_arr_trm,o_orth_arr_trm,
                      u_bnd_arr,v_bnd_arr,dv,ray_scl_arr,ia,DSO,DSD):

    nP, nz, no = vol.shape
    nu = u_bnd_arr.size - 1
    nv = v_bnd_arr.size - 1

    v0 = v_bnd_arr[0]

    for io in range(no):
        p_orth_trm = p_orth_arr_trm[io]
        o_orth_trm = o_orth_arr_trm[io]
        ray_scl = ray_scl_arr[io]

        # 1st-order z projection (same as FP)
        z_bnd_arr_prj_v = z_bnd_arr * DSD / (DSO - o_orth_trm)

        # p-boundaries projected to detector u
        p_bnd_arr_prj_u = proj_img2det_fan(p_drv_bnd_arr_trm, p_orth_trm,
                                           o_drv_bnd_arr_trm, o_orth_trm,
                                           DSO, DSD)

        if p_bnd_arr_prj_u[1] < p_bnd_arr_prj_u[0]:
            print("OH NO")

        # ---- sweep z slabs ----
        for iz in range(nz):

            z_bnd_prj_v_l = z_bnd_arr_prj_v[iz]
            z_bnd_prj_v_r = z_bnd_arr_prj_v[iz + 1]

            iv = int(np.floor((z_bnd_prj_v_l - v0)/dv))
            iv_end = int(np.ceil ((z_bnd_prj_v_r - v0)/dv))

            iv = max(iv, 0)
            iv_end = min(iv_end, nv)

            img_vec = vol[:, io, iz]

            # ---- sweep v ----
            for iv in range(iv, iv_end):

                v_bnd_l = v_bnd_arr[iv]
                v_bnd_r = v_bnd_arr[iv + 1]

                overlap_v_l = max(z_bnd_prj_v_l, v_bnd_l)
                overlap_v_r = min(z_bnd_prj_v_r, v_bnd_r)

                if overlap_v_r <= overlap_v_l:
                    continue

                overlap_v = overlap_v_r - overlap_v_l

                # reset u sweep
                p_bnd_prj_u_l = p_bnd_arr_prj_u[0]
                p_bnd_prj_u_r = p_bnd_arr_prj_u[1]

                u_bnd_l = u_bnd_arr[0]
                u_bnd_r = u_bnd_arr[1]

                ip = 0
                iu = 0

                # ---- fan-style u sweep ----
                while  ip < nP-1 and iu < nu-1:

                    overlap_u_l = max(p_bnd_prj_u_l, u_bnd_l)
                    overlap_u_r = min(p_bnd_prj_u_r, u_bnd_r)

                    if overlap_u_r > overlap_u_l:
                        w = (overlap_u_r - overlap_u_l) * overlap_v * ray_scl
                        img_vec[ip] += sino[ia, iu, iv] * w

                    # advance interval
                    if p_bnd_prj_u_r < u_bnd_r:
                        ip += 1
                        p_bnd_prj_u_l = p_bnd_prj_u_r
                        p_bnd_prj_u_r = p_bnd_arr_prj_u[ip + 1]
                    else:
                        iu += 1
                        u_bnd_l = u_bnd_r
                        u_bnd_r = u_bnd_arr[iu + 1]


@njit(fastmath=True,cache=True)
def _dd_par_geom(img_x, img_y, x_bnd_arr, y_bnd_arr, x_arr, y_arr,
                  cos_ang, sin_ang):
    
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
        ray_scale = 1.0/abs(sin_ang)

        #Rotates the x,y coordinate system to o,p (orthogonal and parallel to
        #the detector)
        #In parallel beam these coordinates will projection directly on the 
        #detector with p(x,y) = -sin_ang*x + cos_ang*y

        # Project pixel boundaries along driving axis onto detector
        # Each pixel x component at the boundary is projected along detetector space
        p_drv_bnd_arr_trm = -sin_ang * x_bnd_arr

        # Each pixel's y component at the center is projected along detetector space
        p_orth_arr_trm = cos_ang * y_arr

        #No transformation need  
        img_trm = img_x

    else:
        # Y-driven
        
        # Scales projected pixel overlap along Y for oblique rays
        ray_scale = 1.0/abs(cos_ang)

        # Project pixel boundaries along driving axis onto detector
        # Each pixel y component at the boundary is projected along detetector space
        p_drv_bnd_arr_trm = cos_ang * y_bnd_arr

        # Each pixel's x component at the center is projected along detetector space
        p_orth_arr_trm = -sin_ang * x_arr

        # Transposes image so axis 0 correspondsto the driving (sweep) axis
        img_trm = img_y 

    # Ensure monotonic increasing for sweep
    if p_drv_bnd_arr_trm[1] < p_drv_bnd_arr_trm[0]:
        p_drv_bnd_arr_trm = p_drv_bnd_arr_trm[::-1]
        img_trm = img_trm[::-1, :]


    # Values are inverted to allow multiplicaion in hot loop

    return img_trm, p_drv_bnd_arr_trm, p_orth_arr_trm, ray_scale


@njit(fastmath=True,cache=True)
def _dd_fan_geom(img_x, img_y, x_bnd_arr, y_bnd_arr, x_arr, y_arr,
                  cos_ang, sin_ang, DSO, DSD):
    
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

        #Rotates the x,y coordinate system to o,p (orthogonal and parallel to
        #the detector)
        #In parallel beam these coordinates will projection directly on the 
        #detector with p(x,y) = -sin_ang*x + cos_ang*y

        # Project pixel boundaries along driving axis onto detector
        # Each pixel x component at the boundary and y component at the center
        #is projected along detetector space
        p_drv_bnd_arr_trm = -sin_ang * x_bnd_arr
        p_orth_arr_trm    =  cos_ang * y_arr

        o_drv_bnd_arr_trm =  cos_ang*x_bnd_arr
        o_orth_arr_trm    =  sin_ang*y_arr

        #No transformation need  
        img_trm = img_x
    else:
        # Y-driven
        
        # Scales projected pixel overlap along Y for oblique rays
        ray_scale = abs(cos_ang)

        # Project pixel boundaries along driving axis onto detector
        # Each pixel y component at the boundary and x component at the center
        #is projected along detetector space
        p_drv_bnd_arr_trm = cos_ang * y_bnd_arr
        p_orth_arr_trm    = -sin_ang * x_arr

        o_drv_bnd_arr_trm = sin_ang*y_bnd_arr
        o_orth_arr_trm    = cos_ang*x_arr

        # Transposes image so axis 0 correspondsto the driving (sweep) axis
        img_trm = img_y 

    # Ensure monotonic increasing for sweep
    if p_drv_bnd_arr_trm[1] < p_drv_bnd_arr_trm[0]:
        p_drv_bnd_arr_trm = p_drv_bnd_arr_trm[::-1]
        o_drv_bnd_arr_trm = o_drv_bnd_arr_trm[::-1]
        
        img_trm = img_trm[::-1,...]


    # Fan-beam  correction:
    r = np.sqrt((DSO - o_orth_arr_trm)**2 + p_orth_arr_trm**2)
    rays_scale = r / ((DSO - o_orth_arr_trm) * ray_scale)
      

    return img_trm, p_drv_bnd_arr_trm, p_orth_arr_trm, o_drv_bnd_arr_trm, o_orth_arr_trm, rays_scale


@njit(fastmath=True,cache=True)
def _dd_fp_cone_geom(img_x, img_y, x_bnd_arr, y_bnd_arr, x_arr, y_arr,
                  cos_ang, sin_ang, DSO, DSD):

    # Determine driving axis
    if abs(sin_ang) >= abs(cos_ang):
        # X-driven
        
        # Scales projected pixel overlap along X for oblique rays
        ray_scale = abs(sin_ang)

        #Rotates the x,y coordinate system to o,p (orthogonal and parallel to
        #the detector)
        #In parallel beam these coordinates will projection directly on the 
        #detector with p(x,y) = -sin_ang*x + cos_ang*y

        # Project pixel boundaries along driving axis onto detector
        # Each pixel x component at the boundary and y component at the center
        #is projected along detetector space
        p_drv_bnd_arr_trm = -sin_ang * x_bnd_arr
        p_orth_arr_trm    =  cos_ang * y_arr

        o_drv_bnd_arr_trm =  cos_ang*x_bnd_arr
        o_orth_arr_trm    =  sin_ang*y_arr

        #No transformation need  
        img_trm = img_x
    else:
        # Y-driven
        
        # Scales projected pixel overlap along Y for oblique rays
        ray_scale = abs(cos_ang)

        # Project pixel boundaries along driving axis onto detector
        # Each pixel y component at the boundary and x component at the center
        #is projected along detetector space
        p_drv_bnd_arr_trm = cos_ang * y_bnd_arr
        p_orth_arr_trm    = -sin_ang * x_arr

        o_drv_bnd_arr_trm = sin_ang*y_bnd_arr
        o_orth_arr_trm    = cos_ang*x_arr

        # Transposes image so axis 0 correspondsto the driving (sweep) axis
        img_trm = img_y 

    # Ensure monotonic increasing for sweep
    if p_drv_bnd_arr_trm[1] < p_drv_bnd_arr_trm[0]:
        p_drv_bnd_arr_trm = p_drv_bnd_arr_trm[::-1]
        o_drv_bnd_arr_trm = o_drv_bnd_arr_trm[::-1]
        
        img_trm = img_trm[::-1,...]


    # Fan-beam  correction:
    r = np.sqrt((DSO - o_orth_arr_trm)**2 + p_orth_arr_trm**2)
    rays_scale = r / ((DSO - o_orth_arr_trm) * ray_scale)
    
    rays_scale = np.ones(p_drv_bnd_arr_trm.size) * (abs(sin_ang) + abs(cos_ang))

    return img_trm, p_drv_bnd_arr_trm, p_orth_arr_trm, o_drv_bnd_arr_trm, o_orth_arr_trm, rays_scale





def dd_fp_par_2d(img,ang_arr,nu,du=1.0,su=0.0,d_pix=1.0):
    img,ang_arr,du,su,d_pix = \
        as_float32(img,ang_arr,du,su,d_pix)
 
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)
   
    return dd_p_par_2d(img,sino,ang_arr,du=du,su=su,d_pix=d_pix,bp=False)


def dd_bp_par_2d(sino,ang_arr,img_shape,du=1.0,su=0.0,d_pix=1.0):
    sino,ang_arr,du,su,d_pix = \
        as_float32(sino,ang_arr,du,su,d_pix)

    img = np.zeros(img_shape, dtype=np.float32)

    return dd_p_par_2d(img,sino,ang_arr,du=du,su=su,d_pix=d_pix,bp=True)


@njit(fastmath=True,cache=True)
def dd_p_par_2d(img,sino,ang_arr,du,su,d_pix,bp):
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
    du : float, optional
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
    na, nu = sino.shape

    #Creates two images where the driving axis is contigious
    #img_x = np.ascontiguousarray(img)
    #img_y = np.ascontiguousarray(img.T)
    img_x = img.copy()
    img_y = img.T.copy()


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
    u_bnd_arr = du*(np.arange(nu + 1, dtype=np.float32) - nu/2.0 + su)

    # Precompute trig functions for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    #Loop through projection angles
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):
        sino_vec = sino[ia]
        
        img_trm, p_drv_bnd_arr_trm, p_orth_arr_trm, ray_scale = \
            _dd_par_geom(img_x,img_y,x_bnd_arr,y_bnd_arr,x_arr,y_arr,
                         cos_ang,sin_ang)
    
        if bp:
            _dd_bp_par_sweep(img_trm,sino_vec,p_drv_bnd_arr_trm,p_orth_arr_trm,
                             u_bnd_arr,ray_scale,ia)
        else:
            _dd_fp_par_sweep(sino_vec,img_trm,p_drv_bnd_arr_trm,p_orth_arr_trm,
                             u_bnd_arr,ray_scale,ia)

    #Return sino normalized with pixel and detector size
    if bp:
        img_y_t = img_y.T.copy()
        return (img_x+img_y_t) / na  / d_pix
    else:    
        return sino*d_pix/du




def dd_fp_fan_2d(img,ang_arr,nu,DSO,DSD,du=1.0,su=0.0,d_pix=1.0):
    img,ang_arr,DSO,DSD,du,su,d_pix = \
        as_float32(img,ang_arr,DSO,DSD,du,su,d_pix)
 
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    return dd_p_fan_2d(img,sino,ang_arr,DSO,DSD,du=du,su=su,d_pix=d_pix,bp=False)


def dd_bp_fan_2d(sino,ang_arr,img_shape,DSO,DSD,du=1.0,su=0.0,d_pix=1.0):
    sino,ang_arr,DSO,DSD,du,su,d_pix = \
        as_float32(sino,ang_arr,DSO,DSD,du,su,d_pix)

    img = np.zeros(img_shape, dtype=np.float32)

    return dd_p_fan_2d(img,sino,ang_arr,DSO,DSD,du=du,su=su,d_pix=d_pix,bp=True)


@njit(fastmath=True,cache=True)
def dd_p_fan_2d(img,sino,ang_arr,DSO,DSD,du,su,d_pix,bp):
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
    na, nu = sino.shape


    #Creates two images where the driving axis is contigious
    #img_x = np.ascontiguousarray(img)
    #img_y = np.ascontiguousarray(img.T)
    img_x = img.copy()
    img_y = img.T.copy()


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
    u_bnd_arr = du*(np.arange(nu + 1, dtype=np.float32) - nu/2.0 + su)

    # Precompute trig functions for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    #Loop through projection angles
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):
        sino_vec = sino[ia]
        
        img_trm, p1_bnd_arr, p2_arr, o1_bnd_arr, o2_arr, ray_scl = \
            _dd_fan_geom(img_x, img_y, x_bnd_arr, y_bnd_arr, x_arr, y_arr,
                 cos_ang, sin_ang, DSO, DSD)


        if bp:
            _dd_bp_fan_sweep(sino_vec,img_trm,p1_bnd_arr,p2_arr,o1_bnd_arr,o2_arr,
                         u_bnd_arr,ray_scl,ia,DSO,DSD)
            
        else:
            _dd_fp_fan_sweep(sino_vec, img_trm, p1_bnd_arr,p2_arr,o1_bnd_arr,o2_arr,
                         u_bnd_arr, ray_scl, ia, DSO, DSD)

    #Return sino normalized with pixel and detector size
    
    if bp:
        img_y_t = img_y.T.copy()
        return (img_x+img_y_t) / na / d_pix 
    else:   
        return sino*d_pix/du



@njit(fastmath=True,cache=True)
def proj_img2det_cone(p, o, z, DSO, DSD):
    t = -DSD / (o - DSO)
    u = t * p
    v = t * z
    return u, v


def dd_fp_cone_3d(img,ang_arr,nu,nv,DSO,DSD,du=1.0,dv=1.0,su=0.0,sv=1.0,d_pix=1.0):
    img,ang_arr,DSO,DSD,du,dv,su,sv,d_pix = \
        as_float32(img,ang_arr,DSO,DSD,du,dv,su,sv,d_pix)
    
    sino = np.zeros((ang_arr.size, nu, nv), dtype=np.float32)

    return dd_p_cone_3d(img,sino,ang_arr,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix,bp=False)



def dd_bp_cone_3d(sino,ang_arr,img_shape,DSO,DSD,du=1.0,dv=1.0,su=0.0,sv=0.0,d_pix=1.0):
    sino,ang_arr,DSO,DSD,du,dv,su,sv,d_pix = \
        as_float32(sino,ang_arr,DSO,DSD,du,dv,su,sv,d_pix)

    img = np.zeros(img_shape, dtype=np.float32)

    return dd_p_cone_3d(img,sino,ang_arr,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix,bp=True)


@njit(fastmath=True,cache=True)
def dd_p_cone_3d(img,sino,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,bp):
    
   
    nx, ny, nz = img.shape
    na, nu, nv = sino.shape

    #img_x = np.ascontiguousarray(img)
    #img_y = np.ascontiguousarray(img.transpose(1, 0, 2))
    img_x = img.copy()
    img_y = img.transpose(1, 0, 2).copy()



    x_bnd_arr = d_pix*(np.arange(nx + 1) - nx/2).astype(np.float32)
    y_bnd_arr = d_pix*(np.arange(ny + 1) - ny/2).astype(np.float32)
    z_bnd_arr = d_pix*(np.arange(nz + 1) - nz/2).astype(np.float32)

    x_arr = (x_bnd_arr[:-1] + x_bnd_arr[1:]) / 2
    y_arr = (y_bnd_arr[:-1] + y_bnd_arr[1:]) / 2
    z_arr = (z_bnd_arr[:-1] + z_bnd_arr[1:]) / 2


    u_bnd_arr = du*(np.arange(nu + 1) - nu/2 + su).astype(np.float32)
    v_bnd_arr = dv*(np.arange(nv + 1) - nv/2 + sv).astype(np.float32)

    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    for ia, (cos_ang, sin_ang) in enumerate(zip(cos_ang_arr, sin_ang_arr)):

        img_trm, p1_bnd_arr, p2_arr, o1_bnd_arr, o2_arr, ray_scl = \
            _dd_fp_cone_geom(img_x, img_y, x_bnd_arr, y_bnd_arr, x_arr, y_arr, 
                             cos_ang, sin_ang, DSO, DSD)      


        
        if bp:
            _dd_bp_cone_sweep(sino, img_trm,p1_bnd_arr, p2_arr, 
                              z_bnd_arr, z_arr, o1_bnd_arr, o2_arr, 
                              u_bnd_arr, v_bnd_arr, dv, ray_scl, ia, DSO, DSD)
        else:
            _dd_fp_cone_sweep(sino, img_trm, p1_bnd_arr, p2_arr, 
                              z_bnd_arr, z_arr, o1_bnd_arr, o2_arr, 
                              u_bnd_arr, v_bnd_arr, dv, ray_scl, ia, DSO, DSD)

    if bp:
        img_y_t = img_y.transpose(1, 0, 2).copy()
        return (img_x+img_y_t) / na / d_pix
    else:
        return sino * (d_pix**2) / (du * dv)





        
 
                
            

















