#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 06:34:23 2026

@author: pvargas21
"""


def _identity_decorator(func):
    return func

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]      # @njit
        return _identity_decorator  # @njit(...)
    
import numpy as np


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

eps = np.float32(1e-6)




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
    
    nP, no = img_trm.shape
    nu = u_bnd_arr.size - 1

    # Loop over orthogonal pixel lines
    for io in range(no):
        
        #Hoisting pointers for explicit LICM
        p_orth_trm = p_orth_arr_trm[io]
        o_orth_trm = o_orth_arr_trm[io]
        ray_scl = ray_scl_arr[io]        

        
        img_vec = img_trm[:, io]
        
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
                if ip==nP: break
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
    
    nP, no = img_trm.shape
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
                if ip==nP: break
                p_bnd_l = p_bnd_r
                p_bnd_r = proj_img2det_fan(p_drv_bnd_arr_trm[ip+1],p_orth_trm,
                                           o_drv_bnd_arr_trm[ip+1],o_orth_trm,
                                           DSO, DSD)
            else:
                iu += 1
                if iu==nu: break
                u_bnd_l = u_bnd_r
                u_bnd_r = u_bnd_arr[iu+1]  






@njit(fastmath=True, cache=True)
def _dd_fp_cone_sweep(sino,vol,p_drv_bnd_arr_trm,p_orth_arr_trm,
                      z_bnd_arr,z_arr,o_drv_bnd_arr_trm,o_orth_arr_trm, 
                      u_bnd_arr,v_bnd_arr,du,dv,ray_scl_arr,ia,DSO,DSD):

    nP, nz, no = vol.shape
    nu = u_bnd_arr.size - 1
    nv = v_bnd_arr.size - 1

    v0 = v_bnd_arr[0]
    u0 = u_bnd_arr[0]
    inv_du = 1.0/du
    inv_dv = 1.0/dv
    
    
    # temporary buffer for u sweep
    tmp_u = np.zeros(nu, dtype=np.float32)

    for io in range(no):
        p_orth_trm = p_orth_arr_trm[io]
        o_orth_trm = o_orth_arr_trm[io]
        ray_scl = ray_scl_arr[io]

        # project z boundaries → v (depends only on io)
        #z_bnd_arr_prj_v = z_bnd_arr * DSD / (DSO - o_orth_trm)
        z_magnification =  DSD/(DSO - o_orth_trm)
        p_magnification =  DSD/(DSO - (o_drv_bnd_arr_trm + o_orth_trm))

        # project p boundaries → u (depends only on io)
        p_bnd_arr_prj_u = p_magnification*(p_drv_bnd_arr_trm + p_orth_trm)
        
        p_u_min = p_bnd_arr_prj_u[0]
        p_u_max = p_bnd_arr_prj_u[-1]

        iu_min = max(int((p_u_min - u0) * inv_du), 0)
        iu_max = min(int((p_u_max - u0) * inv_du) + 1, nu)

        for iz in range(nz):
            z_bnd_prj_v_l = z_magnification*z_bnd_arr[iz]
            z_bnd_prj_v_r = z_magnification*z_bnd_arr[iz + 1]


            iv_min = max(int((z_bnd_prj_v_l - v0) * inv_dv), 0)
            iv_max = min(int((z_bnd_prj_v_r - v0) * inv_dv) + 1, nv)

            if iv_max <= 0 or iv_min >= nv:
                continue

            img_vec = vol[:, io, iz]


            tmp_u[iu_min:iu_max] = 0.0

            ip = 0
            iu = iu_min
            u_bnd_l = u_bnd_arr[iu]
            u_bnd_r = u_bnd_arr[iu + 1]

            p_l = p_bnd_arr_prj_u[0]
            p_r = p_bnd_arr_prj_u[1]

            img_val = img_vec[ip] * ray_scl
            while ip < nP - 1 and iu < iu_max-1:
                overlap_l = max(p_l, u_bnd_l)
                overlap_r = min(p_r, u_bnd_r)

                if overlap_r > overlap_l:
                    tmp_u[iu] += img_val*(overlap_r - overlap_l)

                if p_r < u_bnd_r:
                    ip += 1
                    p_l = p_r
                    p_r = p_bnd_arr_prj_u[ip + 1]
                    img_val = img_vec[ip] * ray_scl
                else:
                    iu += 1
                    u_bnd_l = u_bnd_r
                    u_bnd_r = u_bnd_arr[iu + 1]

            
            for iv in range(iv_min, iv_max):
                v_l = v_bnd_arr[iv]
                v_r = v_bnd_arr[iv + 1]

                ov_l = max(z_bnd_prj_v_l, v_l)
                ov_r = min(z_bnd_prj_v_r, v_r)

                if ov_r <= ov_l:
                    continue


                sino[ia, :, iv] += tmp_u*(ov_r - ov_l)
            
 

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
                while True:

                    overlap_u_l = max(p_bnd_prj_u_l, u_bnd_l)
                    overlap_u_r = min(p_bnd_prj_u_r, u_bnd_r)

                    if overlap_u_r > overlap_u_l:
                        w = (overlap_u_r - overlap_u_l) * overlap_v * ray_scl
                        img_vec[ip] += sino[ia, iu, iv] * w

                    # advance interval
                    if p_bnd_prj_u_r < u_bnd_r:
                        ip += 1
                        if ip==nP: break
                        p_bnd_prj_u_l = p_bnd_prj_u_r
                        p_bnd_prj_u_r = p_bnd_arr_prj_u[ip + 1]
                    else:
                        iu += 1
                        if iu==nu: break
                        u_bnd_l = u_bnd_r
                        u_bnd_r = u_bnd_arr[iu + 1]


@njit(fastmath=True,cache=True)
def _dd_par_geom(img_x, img_y, x_bnd_arr, y_bnd_arr, x_arr, y_arr,
                  cos_ang, sin_ang):
    

    
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
        
        ray_scale = abs(sin_ang)
    else:
        # Y-driven
        
        # Project pixel boundaries along driving axis onto detector
        # Each pixel y component at the boundary and x component at the center
        #is projected along detetector space
        p_drv_bnd_arr_trm = cos_ang * y_bnd_arr
        p_orth_arr_trm    = -sin_ang * x_arr

        o_drv_bnd_arr_trm = sin_ang*y_bnd_arr
        o_orth_arr_trm    = cos_ang*x_arr

        # Transposes image so axis 0 correspondsto the driving (sweep) axis
        img_trm = img_y 
        ray_scale = abs(cos_ang)
        
        
    # Ensure monotonic increasing for sweep
    if p_drv_bnd_arr_trm[1] < p_drv_bnd_arr_trm[0]:
        p_drv_bnd_arr_trm = p_drv_bnd_arr_trm[::-1]
        o_drv_bnd_arr_trm = o_drv_bnd_arr_trm[::-1]
        
        img_trm = img_trm[::-1,...]


    # Fan-beam  correction:
    r = np.sqrt((DSO - o_orth_arr_trm)**2 + p_orth_arr_trm**2)
    rays_scale = r / ((DSO - o_orth_arr_trm) * ray_scale)
    
    #rays_scale = np.ones(p_drv_bnd_arr_trm.size, dtype=np.float32) * (abs(sin_ang) + abs(cos_ang))

    return img_trm, p_drv_bnd_arr_trm, p_orth_arr_trm, o_drv_bnd_arr_trm, o_orth_arr_trm, rays_scale





def dd_fp_par_2d(img,ang_arr,nu,du=1.0,su=0.0,d_pix=1.0,sample=1):
    
    img,ang_arr,du,su,d_pix = \
        as_float32(img,ang_arr,du,su,d_pix)
   
    #if sample != 1:
    #    img = rebin(img, sample *np.array(img.shape))
    #    d_pix /= sample
                     
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)
    return dd_p_par_2d(img,sino,ang_arr,du=du,su=su,d_pix=d_pix,bp=False)




def dd_bp_par_2d(sino,ang_arr,img_shape,du=1.0,su=0.0,d_pix=1.0):
    sino,ang_arr,du,su,d_pix = \
        as_float32(sino,ang_arr,du,su,d_pix)

    img = np.zeros(img_shape, dtype=np.float32)

    return dd_p_par_2d(img,sino,ang_arr,du=du,su=su,d_pix=d_pix,bp=True)


@njit(fastmath=True,cache=True)
def dd_p_par_2d(img,sino,ang_arr,du,su,d_pix,bp):
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


def dd_fp_cone_3d(img,ang_arr,nu,nv,DSO,DSD,du=1.0,dv=1.0,su=0.0,sv=1.0,d_pix=1.0,sample=1):
    img,ang_arr,DSO,DSD,du,dv,su,sv,d_pix = \
        as_float32(img,ang_arr,DSO,DSD,du,dv,su,sv,d_pix)

    if sample != 1:
        img = np.kron(img, np.ones([sample,sample,sample]))
        d_pix /= sample
    
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
                              u_bnd_arr, v_bnd_arr, du, dv, ray_scl, ia, DSO, DSD)

    if bp:
        img_y_t = img_y.transpose(1, 0, 2).copy()
        return (img_x+img_y_t) / na / d_pix
    else:
        return sino * (d_pix**2) / (du * dv)





        
 
                
@njit(fastmath=True, cache=True)
def _dd_fp_square_geom(img,
                       x_bnd_arr, y_bnd_arr, z_bnd_arr,
                       x_arr, y_arr, z_arr,
                       x_s, y_s, z_s,
                       DSD):

    # Driving axis = x (detector normal direction)

    # Distance from source to voxel centers
    o_orth_arr_trm = x_arr - x_s

    # Distance from source to voxel boundaries (driving axis)
    o_drv_bnd_arr_trm = x_bnd_arr - x_s

    # Parallel detector coordinate (u direction = y)
    p_drv_bnd_arr_trm = y_bnd_arr - y_s
    p_orth_arr_trm    = y_arr - y_s

    # No axis swap needed (x driven)
    img_trm = img

    # Cone-beam correction
    r = np.sqrt(o_orth_arr_trm**2 + p_orth_arr_trm**2)
    rays_scale = r / (o_orth_arr_trm)

    return (img_trm,
            p_drv_bnd_arr_trm,
            p_orth_arr_trm,
            o_drv_bnd_arr_trm,
            o_orth_arr_trm,
            rays_scale)








def dd_fp_square_3d(img,
                    src_y_arr,          # translation positions (square edge samples)
                    nu, nv,
                    DSO, DSD,
                    du=1.0, dv=1.0,
                    su=0.0, sv=0.0,
                    d_pix=1.0):

    img, src_y_arr, DSO, DSD, du, dv, su, sv, d_pix = \
        as_float32(img, src_y_arr, DSO, DSD, du, dv, su, sv, d_pix)

    nx, ny, nz = img.shape
    na = src_y_arr.size

    sino = np.zeros((na, nu, nv), dtype=np.float32)

    # Image grids
    x_bnd_arr = d_pix*(np.arange(nx + 1) - nx/2).astype(np.float32)
    y_bnd_arr = d_pix*(np.arange(ny + 1) - ny/2).astype(np.float32)
    z_bnd_arr = d_pix*(np.arange(nz + 1) - nz/2).astype(np.float32)

    x_arr = (x_bnd_arr[:-1] + x_bnd_arr[1:]) / 2
    y_arr = (y_bnd_arr[:-1] + y_bnd_arr[1:]) / 2
    z_arr = (z_bnd_arr[:-1] + z_bnd_arr[1:]) / 2

    # Detector grids
    u_bnd_arr = du*(np.arange(nu + 1) - nu/2 + su).astype(np.float32)
    v_bnd_arr = dv*(np.arange(nv + 1) - nv/2 + sv).astype(np.float32)

    # Source fixed in x (one square edge)
    x_s = -DSO

    for ia in range(na):

        y_s = src_y_arr[ia]

        img_trm, p1_bnd_arr, p2_arr, \
        o1_bnd_arr, o2_arr, ray_scl = \
            _dd_fp_square_geom(img,
                               x_bnd_arr, y_bnd_arr,
                               x_arr, y_arr,
                               x_s, y_s,
                               DSD)

        _dd_fp_cone_sweep(sino, img_trm,
                          p1_bnd_arr, p2_arr,
                          z_bnd_arr, z_arr,
                          o1_bnd_arr, o2_arr,
                          u_bnd_arr, v_bnd_arr,
                          du, dv,
                          ray_scl,
                          ia,
                          DSO, DSD)

    return sino * (d_pix**2) / (du * dv)



from numba import njit
import numpy as np


@njit(fastmath=True, cache=True)
def dd_fp_translational_0deg(
    sino,                # [ty, tz, u, v]
    vol,                 # [nx, ny, nz]
    x_bnd, y_bnd, z_bnd, # voxel boundaries
    u_bnd, v_bnd,        # detector boundaries
    source_x,
    source_y_arr,        # translation in y
    source_z_arr,        # translation in z
    DSD                  # source-detector distance
):

    nx, ny, nz = vol.shape
    nu = u_bnd.size - 1
    nv = v_bnd.size - 1

    u0 = u_bnd[0]
    v0 = v_bnd[0]
    inv_du = 1.0 / (u_bnd[1] - u_bnd[0])
    inv_dv = 1.0 / (v_bnd[1] - v_bnd[0])

    tmp_u = np.zeros(nu, dtype=np.float32)

    for ity in range(source_y_arr.size):
        y_s = source_y_arr[ity]

        for itz in range(source_z_arr.size):
            z_s = source_z_arr[itz]

            for ix in range(nx):

                # Magnification for this x-slice
                denom = source_x - x_bnd[ix]
                if denom <= 0.0:
                    continue  # avoid singularity / behind source

                M = DSD / denom

                # Project z boundaries once for this slice
                z_l = M * (z_bnd[0] - z_s)
                z_r = M * (z_bnd[-1] - z_s)

                iv_min = max(int((z_l - v0) * inv_dv), 0)
                iv_max = min(int((z_r - v0) * inv_dv) + 1, nv)

                if iv_max <= iv_min:
                    continue

                for iz in range(nz):

                    # project this voxel's z extent
                    vz_l = M * (z_bnd[iz] - z_s)
                    vz_r = M * (z_bnd[iz + 1] - z_s)

                    iv0 = max(int((vz_l - v0) * inv_dv), iv_min)
                    iv1 = min(int((vz_r - v0) * inv_dv) + 1, iv_max)

                    if iv1 <= iv0:
                        continue

                    # reset u buffer
                    tmp_u[:] = 0.0

                    # sweep along y (monotonic overlap)
                    ip = 0
                    iu = 0

                    y_l = M * (y_bnd[0] - y_s)
                    y_r = M * (y_bnd[1] - y_s)

                    u_l = u_bnd[0]
                    u_r = u_bnd[1]

                    while ip < ny - 1 and iu < nu - 1:

                        overlap_l = max(y_l, u_l)
                        overlap_r = min(y_r, u_r)

                        if overlap_r > overlap_l:
                            tmp_u[iu] += (
                                vol[ix, ip, iz]
                                * (overlap_r - overlap_l)
                            )

                        if y_r <= u_r:
                            ip += 1
                            y_l = y_r
                            y_r = M * (y_bnd[ip + 1] - y_s)
                        else:
                            iu += 1
                            u_l = u_r
                            u_r = u_bnd[iu + 1]

                    # distribute in v
                    for iv in range(iv0, iv1):

                        v_l = v_bnd[iv]
                        v_r = v_bnd[iv + 1]

                        ov_l = max(vz_l, v_l)
                        ov_r = min(vz_r, v_r)

                        if ov_r > ov_l:
                            sino[ity, itz, :, iv] += tmp_u * (ov_r - ov_l)





@njit(fastmath=True, cache=True)
def dd_fp_translational_0deg_optimized(
    sino,              # [nty, ntz, nv, nu]  (iu contiguous)
    vol,               # [nx, nz, ny]        (iy contiguous)
    x_bnd, y_bnd, z_bnd,
    u_bnd, v_bnd,
    source_x,
    source_y_arr,      # size nty
    source_z_arr,      # size ntz
    DSD
):

    nx, nz, ny = vol.shape
    nty = source_y_arr.size
    ntz = source_z_arr.size
    nu = u_bnd.size - 1
    nv = v_bnd.size - 1

    u0 = u_bnd[0]
    v0 = v_bnd[0]

    inv_du = 1.0 / (u_bnd[1] - u_bnd[0])
    inv_dv = 1.0 / (v_bnd[1] - v_bnd[0])

    tmp_u = np.zeros(nu, dtype=np.float32)

    # ------------------------------------------------------------------
    # OUTERMOST LOOP: x-slice (magnification depends only on x)
    # ------------------------------------------------------------------
    for ix in range(nx):

        denom = source_x - x_bnd[ix]
        if denom <= 0.0:
            continue  # avoid singularity / invalid region

        M = DSD / denom

        # Precompute projected y boundaries once per x-slice
        proj_y_bnd = M * y_bnd

        # Precompute projected z boundaries once per x-slice
        proj_z_bnd = M * z_bnd

        # ------------------------------------------------------------------
        # Loop over z-slices (volume)
        # ------------------------------------------------------------------
        for iz in range(nz):

            # Load contiguous y-column once (stays hot in cache)
            col_y = vol[ix, iz, :]   # contiguous in iy

            # Precompute projected z voxel boundaries
            z_vox_l = proj_z_bnd[iz]
            z_vox_r = proj_z_bnd[iz + 1]

            # ------------------------------------------------------------------
            # Loop over y translations
            # ------------------------------------------------------------------
            for ity in range(nty):

                y_s = source_y_arr[ity]

                # Reset u buffer
                tmp_u[:] = 0.0

                # Monotonic overlap sweep in y
                ip = 0
                iu = 0

                y_l = proj_y_bnd[0] - M * y_s
                y_r = proj_y_bnd[1] - M * y_s

                u_l = u_bnd[0]
                u_r = u_bnd[1]

                while ip < ny - 1 and iu < nu - 1:

                    overlap_l = max(y_l, u_l)
                    overlap_r = min(y_r, u_r)

                    if overlap_r > overlap_l:
                        tmp_u[iu] += col_y[ip] * (overlap_r - overlap_l)

                    if y_r <= u_r:
                        ip += 1
                        y_l = y_r
                        y_r = proj_y_bnd[ip + 1] - M * y_s
                    else:
                        iu += 1
                        u_l = u_r
                        u_r = u_bnd[iu + 1]

                # ------------------------------------------------------------------
                # Now reuse tmp_u across all z translations
                # ------------------------------------------------------------------
                for itz in range(ntz):

                    z_s = source_z_arr[itz]

                    vz_l = z_vox_l - M * z_s
                    vz_r = z_vox_r - M * z_s

                    iv_min = int((vz_l - v0) * inv_dv)
                    iv_max = int((vz_r - v0) * inv_dv) + 1

                    if iv_max <= 0 or iv_min >= nv:
                        continue

                    if iv_min < 0:
                        iv_min = 0
                    if iv_max > nv:
                        iv_max = nv

                    for iv in range(iv_min, iv_max):

                        v_l = v_bnd[iv]
                        v_r = v_bnd[iv + 1]

                        ov_l = max(vz_l, v_l)
                        ov_r = min(vz_r, v_r)

                        if ov_r > ov_l:
                            # contiguous write in iu
                            sino[ity, itz, iv, :] += tmp_u * (ov_r - ov_l)



@njit(fastmath=True, cache=True)
def dd_bp_translational_0deg_optimized(
    vol,
    sino,
    x_bnd, y_bnd, z_bnd,
    u_bnd, v_bnd,
    source_x,
    source_y_arr,
    source_z_arr,
    DSD
):

    nx, nz, ny = vol.shape
    nty, ntz, nv, nu = sino.shape

    u0 = u_bnd[0]
    v0 = v_bnd[0]

    inv_du = 1.0 / (u_bnd[1] - u_bnd[0])
    inv_dv = 1.0 / (v_bnd[1] - v_bnd[0])

    # --------------------------------------------------
    # Loop over x slices (magnification depends only on x)
    # --------------------------------------------------
    for ix in range(nx):

        denom = source_x - x_bnd[ix]
        if denom <= 0.0:
            continue

        M = DSD / denom

        proj_y_bnd = M * y_bnd
        proj_z_bnd = M * z_bnd

        # --------------------------------------------------
        for iz in range(nz):

            z_l0 = proj_z_bnd[iz]
            z_r0 = proj_z_bnd[iz + 1]

            for iy in range(ny):

                y_l0 = proj_y_bnd[iy]
                y_r0 = proj_y_bnd[iy + 1]

                acc = 0.0

                # Loop over translations
                for ity in range(nty):

                    y_s = source_y_arr[ity]
                    y_l = y_l0 - M * y_s
                    y_r = y_r0 - M * y_s

                    iu_min = int((y_l - u0) * inv_du)
                    iu_max = int((y_r - u0) * inv_du) + 1

                    if iu_max <= 0 or iu_min >= nu:
                        continue

                    if iu_min < 0:
                        iu_min = 0
                    if iu_max > nu:
                        iu_max = nu

                    for itz in range(ntz):

                        z_s = source_z_arr[itz]
                        z_l = z_l0 - M * z_s
                        z_r = z_r0 - M * z_s

                        iv_min = int((z_l - v0) * inv_dv)
                        iv_max = int((z_r - v0) * inv_dv) + 1

                        if iv_max <= 0 or iv_min >= nv:
                            continue

                        if iv_min < 0:
                            iv_min = 0
                        if iv_max > nv:
                            iv_max = nv

                        # distance-driven overlap
                        for iv in range(iv_min, iv_max):

                            v_l = v_bnd[iv]
                            v_r = v_bnd[iv + 1]

                            ov_v = min(z_r, v_r) - max(z_l, v_l)
                            if ov_v <= 0:
                                continue

                            row = sino[ity, itz, iv, :]

                            for iu in range(iu_min, iu_max):

                                u_l = u_bnd[iu]
                                u_r = u_bnd[iu + 1]

                                ov_u = min(y_r, u_r) - max(y_l, u_l)
                                if ov_u > 0:
                                    acc += row[iu] * ov_u * ov_v

                vol[ix, iz, iy] += acc