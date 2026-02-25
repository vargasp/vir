#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 06:34:23 2026

@author: pvargas21
"""


def _identity_decorator(func):
    return func

try:
    from numba import njit, prange
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]      # @njit
        return _identity_decorator  # @njit(...)
    
import numpy as np
import vir.sys_mat.pf as pf

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
            while True:
                overlap_l = max(p_l, u_bnd_l)
                overlap_r = min(p_r, u_bnd_r)

                if overlap_r > overlap_l:
                    tmp_u[iu] += img_val*(overlap_r - overlap_l)

                if p_r < u_bnd_r:
                    ip += 1
                    if ip == nP:
                        break
                    p_l = p_r
                    p_r = p_bnd_arr_prj_u[ip + 1]
                    img_val = img_vec[ip] * ray_scl
                else:
                    iu += 1
                    if iu == nu:
                        break
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





        
 
                






def dd_p_square(img,nu,nv,ns_p,ns_z,DSO,DSD,
                  du=1.0,dv=1.0,dsrc_p=1.0,dsrc_z=1.0, 
                  su=0.0,sv=0.0,d_pix=1.0):

    nx, ny, nz = img.shape
    
    if nx != ny:
        raise ValueError("nx must equal ny")

    
    sino = np.zeros([ns_p,ns_z,nv,nu,4], np.float32)

    img,DSO,DSD,du,dv,dsrc_p,dsrc_z,d_pix = \
        as_float32(img,DSO,DSD,du,dv,dsrc_p,dsrc_z,d_pix)



    #[ny, nz, nx]  (iy contiguous)
    imgX = np.ascontiguousarray(img.transpose(1,2,0))
    
    #[nx, nz, ny]  (iy contiguous)
    imgY = np.ascontiguousarray(img.transpose(0,2,1))
    

    sino = dd_p_square_num(imgX,imgY,sino,DSO,DSD,du,dv,dsrc_p,dsrc_z,su,sv,d_pix)
    
    #return np.ascontiguousarray(sino.transpose(0,1,2,4,3))
    return sino


@njit(fastmath=True, cache=True)
def dd_p_square_num(imgX,imgY,sino,DSO,DSD,du,dv,dsrc_p,dsrc_z,su,sv,d_pix):   
    no, nz, np = imgX.shape


    #ns, nsrc_p, nsrc_z, nv, nu = sino.shape    
    nsrc_p, nsrc_z, nv, nu, ns = sino.shape    
  
    # voxel boundaries in parallel (p), orthogonal (o), and vertical (z)
    p_bnd_arr = pf.boundspace(d_pix, np)  # parallel
    o_bnd_arr = pf.boundspace(d_pix, no)  # orthogonal
    z_bnd_arr = pf.boundspace(d_pix, nz)  # vertical

    # Detector grids  
    u_bnd_arr = pf.boundspace(du,nu,su)
    v_bnd_arr = pf.boundspace(dv,nv,sv)

    # Source coordinates
    src_p_arr = pf.censpace(dsrc_p,nsrc_p)
    src_z_arr = pf.censpace(dsrc_z,nsrc_z)

    dd_fp_translational(
        sino,                # [4,ty, tz, u, v]
        imgY,imgX,          # [nx, nz, ny] , [ny, nz, nx] 
        p_bnd_arr, o_bnd_arr, z_bnd_arr, # voxel boundaries
        u_bnd_arr, v_bnd_arr,du,dv,        # detector boundaries
        DSO,
        src_p_arr,        # translation in y
        src_z_arr,        # translation in z
        DSD                  # source-detector distance
    )

    return sino * (d_pix**2) / (du * dv)




                                
                
@njit(fastmath=True, parallel=True, cache=True)
def dd_fp_translational(
    sino,              # [nsrc_p, nsrc_z, nv, 4, nu]
    imgY, imgX,        # [no, nz, nP]
    p_bnd_arr, o_bnd_arr, z_bnd_arr,
    u_bnd, v_bnd, du, dv,
    src_o,src_p_arr,src_z_arr,
    DSD
):
    no, nz, nP = imgY.shape
    nsrc_p, nsrc_z, nv, nu, _ = sino.shape

    inv_du = np.float32(1.0) / du
    inv_dv = np.float32(1.0) / dv

    u0 = u_bnd[0]
    v0 = v_bnd[0]

    M_arr = DSD /(src_o - o_bnd_arr)

    # PARALLEL OVER ORTHOGONAL SLICES
    for io in prange(no):

        M = M_arr[io]
        
        proj_p_bnd_arr = M * p_bnd_arr
        proj_z_bnd_arr = M * z_bnd_arr
        proj_src_p_arr = M * src_p_arr
        proj_src_z_arr = M * src_z_arr
        
        # Thread-local buffer
        tmp_u = np.zeros((nu,4), dtype=np.float32)
        u_lo = nu
        u_hi = 0
                
        # Loop over z slices
        for iz in range(nz):

            colX  = imgX[io, iz, :]
            colY  = imgY[io, iz, :]
            colXF = imgX[no - 1 - io, iz, :]
            colYF = imgY[no - 1 - io, iz, :]

            vz_l_arr = proj_z_bnd_arr[iz] - proj_src_z_arr
            vz_r_arr = proj_z_bnd_arr[iz + 1] - proj_src_z_arr
            
            # vz_l_arr and vz_r_arr are arrays of length nsrc_z (float32)
            iv_min_arr = np.clip(((vz_l_arr - v0) * inv_dv).astype(np.int32), 0, nv)
            iv_max_arr = np.clip(((vz_r_arr - v0) * inv_dv).astype(np.int32) + 1, 0, nv)

            # Loop over parallel source
            for i_sp in range(nsrc_p):

                #Initialize temp array
                for iu in range(u_lo, u_hi):
                    for f in range(4):
                        tmp_u[iu,f] = 0.0

                u_lo = nu
                u_hi = 0

                proj_src_p = proj_src_p_arr[i_sp]
               

                # Loop over parallel voxels
                for ip in range(nP):

                    p_l = proj_p_bnd_arr[ip]     - proj_src_p
                    p_r = proj_p_bnd_arr[ip + 1] - proj_src_p

                    iu_min = int((p_l - u0)*inv_du)
                    iu_max = int((p_r - u0)*inv_du) + 1

                    iu_min = max(0, iu_min)
                    iu_max = min(nu, iu_max)                      

                    if iu_min >= iu_max:
                        continue

                    v0f = colY[ip]
                    v1f = colX[nP - 1 - ip]
                    v2f = colYF[nP - 1 - ip]
                    v3f = colXF[ip]
                    
                    u_lo = min(u_lo,iu_min)
                    u_hi = max(u_hi,iu_max)

                    # Loop over u detectors
                    for iu in range(iu_min, iu_max):
                        left  = p_l if p_l > u_bnd[iu] else u_bnd[iu]
                        right = p_r if p_r < u_bnd[iu+1] else u_bnd[iu+1]
                        overlap_u = right - left

                        if overlap_u > 0.0:
                            tmp_u[iu,0] += v0f * overlap_u
                            tmp_u[iu,1] += v1f * overlap_u
                            tmp_u[iu,2] += v2f * overlap_u
                            tmp_u[iu,3] += v3f * overlap_u

                # -------- Z integration --------
                for i_sz in range(nsrc_z):
                    iv_min = iv_min_arr[i_sz]
                    iv_max = iv_max_arr[i_sz]
                    
                    if iv_min >= iv_max:
                        continue

                    vz_l = vz_l_arr[i_sz]
                    vz_r = vz_r_arr[i_sz]
                    
                    base = sino[i_sp, i_sz]  # hoist partial indexing

                    for iu in range(u_lo, u_hi):
                    
                        # Keep tmp_u values in registers
                        t0 = tmp_u[iu, 0]
                        t1 = tmp_u[iu, 1]
                        t2 = tmp_u[iu, 2]
                        t3 = tmp_u[iu, 3]
                    
                        # Skip if all zero (optional micro-optimization)
                        if t0 == 0.0 and t1 == 0.0 and t2 == 0.0 and t3 == 0.0:
                            continue
                    
                        for iv in range(iv_min, iv_max):
                            v_l = v_bnd[iv]
                            v_r = v_bnd[iv + 1]
                    
                            left  = vz_l if vz_l > v_l else v_l
                            right = vz_r if vz_r < v_r else v_r
                            overlap_v = right - left
                    
                            if overlap_v > 0.0:                   
                                # contiguous in last dim (f)
                                base[iv, iu, 0] += t0 * overlap_v
                                base[iv, iu, 1] += t1 * overlap_v
                                base[iv, iu, 2] += t2 * overlap_v
                                base[iv, iu, 3] += t3 * overlap_v
                                    
                                    
                                    
                                    
                                    
def dd_bp_square(sino,img_shape,DSO,DSD,
                  du=1.0,dv=1.0,dsrc_p=1.0,dsrc_z=1.0, 
                  su=0.0,sv=0.0,d_pix=1.0):

    nx, ny, nz = img_shape
    
    if nx != ny:
        raise ValueError("nx must equal ny")

    
    img = np.zeros(img_shape, dtype=np.float32)

    sino,DSO,DSD,du,dv,dsrc_p,dsrc_z,d_pix = \
        as_float32(sino,DSO,DSD,du,dv,dsrc_p,dsrc_z,d_pix)

    sino = np.ascontiguousarray(sino.transpose(0,1,2,4,3))

    vol = dd_bp_square_num(img,sino,DSO,DSD,du,dv,dsrc_p,dsrc_z,su,sv,d_pix)
    
    return vol


@njit(fastmath=True, cache=True)
def dd_bp_square_num(vol,sino,DSO,DSD,du,dv,dsrc_p,dsrc_z,su,sv,d_pix):   
    no, nz, np = vol.shape


    #ns, nsrc_p, nsrc_z, nv, nu = sino.shape    
    nsrc_p, nsrc_z, nv, nu, ns = sino.shape    
  
    # voxel boundaries in parallel (p), orthogonal (o), and vertical (z)
    p_bnd_arr = pf.boundspace(d_pix, np)  # parallel
    o_bnd_arr = pf.boundspace(d_pix, no)  # orthogonal
    z_bnd_arr = pf.boundspace(d_pix, nz)  # vertical

    # Detector grids  
    u_bnd_arr = pf.boundspace(du,nu,su)
    v_bnd_arr = pf.boundspace(dv,nv,sv)

    # Source coordinates
    src_p_arr = pf.censpace(dsrc_p,nsrc_p)
    src_z_arr = pf.censpace(dsrc_z,nsrc_z)

    dd_bp_translational(
        sino,                # [4,ty, tz, u, v]
        vol,          # [nx, nz, ny] , [ny, nz, nx] 
        p_bnd_arr, o_bnd_arr, z_bnd_arr, # voxel boundaries
        u_bnd_arr, v_bnd_arr,du,dv,        # detector boundaries
        DSO,
        src_p_arr,        # translation in y
        src_z_arr,        # translation in z
        DSD                  # source-detector distance
    )

    return vol
                                    


@njit(fastmath=True, parallel=True, cache=True)
def dd_bp_translational(
    sino,              # [nsrc_p, nsrc_z, nv, 4, nu]  (INPUT)
    vol,               # [no, nz, nP]                (OUTPUT)
    p_bnd_arr, o_bnd_arr, z_bnd_arr,
    u_bnd, v_bnd, du, dv,
    src_o, src_p_arr, src_z_arr,
    DSD
):
    no, nz, nP = vol.shape
    nsrc_p, nsrc_z, nv, _, nu = sino.shape

    inv_du = np.float32(1.0) / du
    inv_dv = np.float32(1.0) / dv

    u0 = u_bnd[0]
    v0 = v_bnd[0]

    # Precompute magnification
    M_arr = DSD / (src_o - o_bnd_arr)

    # --------------------------------------------
    # Parallel over orthogonal slices
    # --------------------------------------------
    for io in prange(no):

        M = M_arr[io]
        if M <= 0.0:
            continue

        proj_p_bnd = M * p_bnd_arr
        proj_z_bnd = M * z_bnd_arr
        proj_src_p = M * src_p_arr
        proj_src_z = M * src_z_arr

        for iz in range(nz):

            vz_l_arr = proj_z_bnd[iz]     - proj_src_z
            vz_r_arr = proj_z_bnd[iz + 1] - proj_src_z

            iv_min_arr = np.clip(((vz_l_arr - v0) * inv_dv).astype(np.int32), 0, nv)
            iv_max_arr = np.clip(((vz_r_arr - v0) * inv_dv).astype(np.int32) + 1, 0, nv)

            # Get 2D volume slice (contiguous in last dim)
            vol_slice = vol[io, iz]

            # ----------------------------------------
            # Loop over sources
            # ----------------------------------------
            for i_sp in range(nsrc_p):

                src_p_val = proj_src_p[i_sp]

                # ----------------------------------------
                # Loop over voxels in p-direction
                # ----------------------------------------
                for ip in range(nP):

                    p_l = proj_p_bnd[ip]     - src_p_val
                    p_r = proj_p_bnd[ip + 1] - src_p_val

                    iu_min = int((p_l - u0) * inv_du)
                    iu_max = int((p_r - u0) * inv_du) + 1

                    if iu_min < 0:
                        iu_min = 0
                    if iu_max > nu:
                        iu_max = nu
                    if iu_min >= iu_max:
                        continue

                    voxel_accum = 0.0

                    # ----------------------------------------
                    # Integrate over detector u
                    # ----------------------------------------
                    for iu in range(iu_min, iu_max):

                        left  = p_l if p_l > u_bnd[iu] else u_bnd[iu]
                        right = p_r if p_r < u_bnd[iu+1] else u_bnd[iu+1]
                        overlap_u = right - left

                        if overlap_u <= 0.0:
                            continue

                        # ----------------------------------------
                        # Integrate over detector v
                        # ----------------------------------------
                        for i_sz in range(nsrc_z):

                            iv_min = iv_min_arr[i_sz]
                            iv_max = iv_max_arr[i_sz]
                            if iv_min >= iv_max:
                                continue

                            vz_l = vz_l_arr[i_sz]
                            vz_r = vz_r_arr[i_sz]

                            sino_block = sino[i_sp, i_sz]

                            for iv in range(iv_min, iv_max):

                                v_l = v_bnd[iv]
                                v_r = v_bnd[iv+1]

                                left_v  = vz_l if vz_l > v_l else v_l
                                right_v = vz_r if vz_r < v_r else v_r
                                overlap_v = right_v - left_v

                                if overlap_v > 0.0:
                                    # collapse 4 detector channels
                                    s0 = sino_block[iv, 0, iu]
                                    s1 = sino_block[iv, 1, iu]
                                    s2 = sino_block[iv, 2, iu]
                                    s3 = sino_block[iv, 3, iu]

                                    voxel_accum += (
                                        (s0 + s1 + s2 + s3)
                                        * overlap_u
                                        * overlap_v
                                    )

                    # accumulate into volume voxel
                    vol_slice[ip] += voxel_accum          
                                    
                                    








"""
@njit(fastmath=True, parallel=True, cache=True)
def dd_fp_translational_tiled(
    sino,              # [nsrc_p, nsrc_z, nv, 4, nu]
    imgY, imgX,        # [no, nz, nP]
    p_bnd_arr, o_bnd_arr, z_bnd_arr,
    u_bnd, v_bnd, du, dv,
    src_o, src_p_arr, src_z_arr,
    DSD           # <--- tunable tile size
):
    
    
    tile_nu  = 32        # <--- tunable tile size

    no, nz, nP = imgY.shape
    nsrc_p, nsrc_z, nv, nu, _ = sino.shape

    inv_du = 1.0 / du
    inv_dv = 1.0 / dv

    u0 = u_bnd[0]
    v0 = v_bnd[0]

    M_arr = DSD / (src_o - o_bnd_arr)

    # Parallel over orthogonal slices
    for io in prange(no):

        M = M_arr[io]
        if M <= 0.0:
            continue

        proj_p_bnd_arr = M * p_bnd_arr
        proj_z_bnd_arr = M * z_bnd_arr
        proj_src_p_arr = M * src_p_arr
        proj_src_z_arr = M * src_z_arr

        # Thread-local buffer (nu × 4)
        tmp_u = np.zeros((nu, 4), dtype=np.float32)

        # Loop over z slices
        for iz in range(nz):

            colX  = imgX[io, iz, :]
            colY  = imgY[io, iz, :]
            colXF = imgX[no - 1 - io, iz, :]
            colYF = imgY[no - 1 - io, iz, :]

            vz_l_arr = proj_z_bnd_arr[iz] - proj_src_z_arr
            vz_r_arr = proj_z_bnd_arr[iz + 1] - proj_src_z_arr

            iv_min_arr = np.clip(((vz_l_arr - v0) * inv_dv).astype(np.int32), 0, nv)
            iv_max_arr = np.clip(((vz_r_arr - v0) * inv_dv).astype(np.int32) + 1, 0, nv)

            # Loop over parallel sources
            for i_sp in range(nsrc_p):

                proj_src_p = proj_src_p_arr[i_sp]

                # Compute iu_min and iu_max for this source
                u_lo = nu
                u_hi = 0
                for ip in range(nP):
                    p_l = proj_p_bnd_arr[ip] - proj_src_p
                    p_r = proj_p_bnd_arr[ip+1] - proj_src_p

                    iu_min = max(0, int((p_l - u0) * inv_du))
                    iu_max = min(nu, int((p_r - u0) * inv_du) + 1)

                    if iu_min >= iu_max:
                        continue
                    u_lo = min(u_lo, iu_min)
                    u_hi = max(u_hi, iu_max)

                # Reset tmp_u only in the active tile range
                for iu in range(u_lo, u_hi):
                    for f in range(4):
                        tmp_u[iu,f] = 0.0

                # -------- Parallel integration over p --------
                for ip in range(nP):

                    p_l = proj_p_bnd_arr[ip] - proj_src_p
                    p_r = proj_p_bnd_arr[ip+1] - proj_src_p

                    iu_min = max(0, int((p_l - u0) * inv_du))
                    iu_max = min(nu, int((p_r - u0) * inv_du) + 1)

                    if iu_min >= iu_max:
                        continue

                    v0f = colY[ip]
                    v1f = colX[nP - 1 - ip]
                    v2f = colYF[nP - 1 - ip]
                    v3f = colXF[ip]

                    # Tile over iu
                    for tile_start in range(iu_min, iu_max, tile_nu):
                        tile_end = min(iu_max, tile_start + tile_nu)

                        for iu in range(tile_start, tile_end):
                            left  = max(p_l, u_bnd[iu])
                            right = min(p_r, u_bnd[iu+1])
                            overlap_u = right - left
                            if overlap_u > 0.0:
                                tmp_u[iu,0] += v0f * overlap_u
                                tmp_u[iu,1] += v1f * overlap_u
                                tmp_u[iu,2] += v2f * overlap_u
                                tmp_u[iu,3] += v3f * overlap_u

                # -------- Z integration --------
                for i_sz in range(nsrc_z):
                    iv_min = iv_min_arr[i_sz]
                    iv_max = iv_max_arr[i_sz]

                    if iv_min >= iv_max:
                        continue

                    vz_l = vz_l_arr[i_sz]
                    vz_r = vz_r_arr[i_sz]

                    base = sino[i_sp, i_sz]

                    # Tile over iu for cache reuse
                    for tile_start in range(u_lo, u_hi, tile_nu):
                        tile_end = min(u_hi, tile_start + tile_nu)

                        for iu in range(tile_start, tile_end):
                            t0 = tmp_u[iu,0]
                            t1 = tmp_u[iu,1]
                            t2 = tmp_u[iu,2]
                            t3 = tmp_u[iu,3]

                            if t0 == 0.0 and t1 == 0.0 and t2 == 0.0 and t3 == 0.0:
                                continue

                            for iv in range(iv_min, iv_max):
                                v_l = v_bnd[iv]
                                v_r = v_bnd[iv+1]
                                overlap_v = max(0.0, min(vz_r, v_r) - max(vz_l, v_l))
                                if overlap_v > 0.0:
                                    base[iv, iu, 0] += t0 * overlap_v
                                    base[iv, iu, 1] += t1 * overlap_v
                                    base[iv, iu, 2] += t2 * overlap_v
                                    base[iv, iu, 3] += t3 * overlap_v
                                    
                                    
                                    
                                    
                                    

from numba import cuda, float32, int32
import numpy as np

@cuda.jit
def dd_fp_forward_gpu_tiled(
    sino,           # [nsrc_p, nsrc_z, nv, nu, 4] float32
    imgY, imgX,     # [no, nz, nP] float32
    p_bnd_arr, o_bnd_arr, z_bnd_arr,
    u_bnd, v_bnd,
    source_o,
    src_p_arr,
    src_z_arr,
    DSD
):
    # -------------------------------------
    # Grid mapping: (io, iz) per block, threads along i_sp, i_sz
    # -------------------------------------
    io, iz = cuda.blockIdx.x, cuda.blockIdx.y
    i_sp, i_sz = cuda.threadIdx.x, cuda.threadIdx.y

    no, nz, nP = imgY.shape
    nsrc_p, nsrc_z, nv, nu, _ = sino.shape

    if io >= no or iz >= nz or i_sp >= nsrc_p or i_sz >= nsrc_z:
        return

    # -------------------------------------
    # Compute magnification for this slice
    denom = source_o - o_bnd_arr[io]
    if denom <= 0.0:
        return
    M = DSD / denom

    # Precompute projected bounds in thread-local memory
    proj_p_bnd = cuda.local.array(shape=1024, dtype=float32)  # adjust to nP+1
    proj_z_bnd = cuda.local.array(shape=64, dtype=float32)    # adjust to nz+1
    proj_src_p = M * src_p_arr[i_sp]

    for ip in range(nP + 1):
        proj_p_bnd[ip] = M * p_bnd_arr[ip]

    for izb in range(nz + 1):
        proj_z_bnd[izb] = M * z_bnd_arr[izb]

    # -------------------------------------
    # Shared memory for tmp_u accumulation
    # shape: tile_nu × 4
    tile_nu = 128  # tune depending on GPU
    tmp_u = cuda.shared.array(shape=(128,4), dtype=float32)

    # initialize shared tmp_u to zero
    for iu in range(tile_nu):
        for f in range(4):
            tmp_u[iu,f] = 0.0
    cuda.syncthreads()

    # Load voxel column
    colX  = imgX[io, iz, :]
    colY  = imgY[io, iz, :]
    colXF = imgX[no - 1 - io, iz, :]
    colYF = imgY[no - 1 - io, iz, :]

    u0 = u_bnd[0]
    du = u_bnd[1] - u_bnd[0]
    inv_du = 1.0 / du
    v0 = v_bnd[0]
    dv = v_bnd[1] - v_bnd[0]
    inv_dv = 1.0 / dv

    # -------------------------------------
    # Distance-driven integration along "p" (u-axis)
    u_lo = nu
    u_hi = 0
    for ip in range(nP):
        p_l = proj_p_bnd[ip] - proj_src_p
        p_r = proj_p_bnd[ip+1] - proj_src_p

        iu_min = int((p_l - u0) * inv_du)
        iu_max = int((p_r - u0) * inv_du) + 1
        iu_min = max(0, iu_min)
        iu_max = min(nu, iu_max)

        if iu_min >= iu_max:
            continue

        v0f = colY[ip]
        v1f = colX[nP - 1 - ip]
        v2f = colYF[nP - 1 - ip]
        v3f = colXF[ip]

        if iu_min < u_lo:
            u_lo = iu_min
        if iu_max > u_hi:
            u_hi = iu_max

        # accumulate into shared memory
        for iu in range(iu_min, iu_max):
            left  = max(p_l, u_bnd[iu])
            right = min(p_r, u_bnd[iu+1])
            ov_u = right - left
            if ov_u > 0.0:
                tmp_u[iu,f] += v0f*ov_u if f==0 else 0
                tmp_u[iu,f] += v1f*ov_u if f==1 else 0
                tmp_u[iu,f] += v2f*ov_u if f==2 else 0
                tmp_u[iu,f] += v3f*ov_u if f==3 else 0

    cuda.syncthreads()  # make sure tmp_u is ready for all threads in block

    # -------------------------------------
    # Distance-driven integration along "z" (v-axis)
    vz_l = proj_z_bnd[iz] - M * src_z_arr[i_sz]
    vz_r = proj_z_bnd[iz+1] - M * src_z_arr[i_sz]

    iv_min = int((vz_l - v0) * inv_dv)
    iv_max = int((vz_r - v0) * inv_dv) + 1
    iv_min = max(0, iv_min)
    iv_max = min(nv, iv_max)

    for iv in range(iv_min, iv_max):
        left  = max(vz_l, v_bnd[iv])
        right = min(vz_r, v_bnd[iv+1])
        ov_v = right - left
        if ov_v <= 0.0:
            continue

        # coalesced writes to global memory
        for iu in range(u_lo, u_hi):
            for f in range(4):
                sino[i_sp, i_sz, iv, iu, f] += tmp_u[iu,f] * ov_v





"""















@njit(fastmath=True, parallel=True, cache=True)
def dd_bp_translational_opt_prange(
    imgY, imgX,          # [no, nz, nP]  (output, modified in-place)
    sino,                # [nsrc_p, nsrc_z, nv, 4, nu]
    p_bnd_arr, o_bnd_arr, z_bnd_arr,
    u_bnd, v_bnd,
    source_o,
    src_p_arr,
    src_z_arr,
    DSD
):

    no, nz, nP = imgY.shape
    nsrc_p, nsrc_z, nv, _, nu = sino.shape

    du = u_bnd[1] - u_bnd[0]
    inv_dv = np.float32(1.0) / (v_bnd[1] - v_bnd[0])
    v0 = v_bnd[0]

    # ------------------------------------------------------------
    # Precompute magnification per orthogonal slice
    # ------------------------------------------------------------
    M_arr = np.empty(no, dtype=np.float32)
    for io in range(no):
        denom = source_o - o_bnd_arr[io]
        if denom > 0.0:
            M_arr[io] = DSD / denom
        else:
            M_arr[io] = np.float32(0.0)

    # ============================================================
    # PARALLEL OVER ORTHOGONAL SLICES (SAFE)
    # ============================================================
    for io in prange(no):

        M = M_arr[io]
        if M == 0.0:
            continue

        proj_p_bnd = M * p_bnd_arr
        proj_z_bnd = M * z_bnd_arr

        # Local image slices (thread owned)
        imgY_slice  = imgY[io, :, :]
        imgX_slice  = imgX[io, :, :]
        imgYF_slice = imgY[no - 1 - io, :, :]
        imgXF_slice = imgX[no - 1 - io, :, :]

        # --------------------------------------------------------
        # Loop over z slices
        # --------------------------------------------------------
        for iz in range(nz):

            z_vox_l = proj_z_bnd[iz]
            z_vox_r = proj_z_bnd[iz + 1]

            for i_sp in range(nsrc_p):

                Mp = M * src_p_arr[i_sp]

                # ------------------------------------------------
                # Parallel direction (u sweep)
                # ------------------------------------------------
                for ip in range(nP):

                    p_l = proj_p_bnd[ip]     - Mp
                    p_r = proj_p_bnd[ip + 1] - Mp

                    iu_min = int((p_l - u_bnd[0]) / du)
                    iu_max = int((p_r - u_bnd[0]) / du) + 1

                    if iu_min < 0:
                        iu_min = 0
                    if iu_max > nu:
                        iu_max = nu
                    if iu_min >= iu_max:
                        continue

                    # accumulate per face
                    acc0 = np.float32(0.0)
                    acc1 = np.float32(0.0)
                    acc2 = np.float32(0.0)
                    acc3 = np.float32(0.0)

                    for iu in range(iu_min, iu_max):

                        left  = p_l if p_l > u_bnd[iu] else u_bnd[iu]
                        right = p_r if p_r < u_bnd[iu+1] else u_bnd[iu+1]
                        ov = right - left

                        if ov > 0.0:

                            # integrate over z translations
                            for i_sz in range(nsrc_z):

                                vz_l = z_vox_l - M * src_z_arr[i_sz]
                                vz_r = z_vox_r - M * src_z_arr[i_sz]

                                iv_min = int((vz_l - v0) * inv_dv)
                                iv_max = int((vz_r - v0) * inv_dv) + 1

                                if iv_min < 0:
                                    iv_min = 0
                                if iv_max > nv:
                                    iv_max = nv
                                if iv_min >= iv_max:
                                    continue

                                for iv in range(iv_min, iv_max):

                                    left_v  = vz_l if vz_l > v_bnd[iv] else v_bnd[iv]
                                    right_v = vz_r if vz_r < v_bnd[iv+1] else v_bnd[iv+1]
                                    ov_v = right_v - left_v

                                    if ov_v > 0.0:

                                        w = ov * ov_v

                                        s = sino[i_sp, i_sz, iv, :, iu]

                                        acc0 += s[0] * w
                                        acc1 += s[1] * w
                                        acc2 += s[2] * w
                                        acc3 += s[3] * w

                    # ------------------------------------------------
                    # Write accumulated values to image (adjoint)
                    # ------------------------------------------------
                    imgY_slice[iz, ip]               += acc0
                    imgX_slice[iz, nP - 1 - ip]      += acc1
                    imgYF_slice[iz, nP - 1 - ip]     += acc2
                    imgXF_slice[iz, ip]              += acc3
                    
                    
                    

"""
#@njit(fastmath=True, cache=True)
def dd_fp_translational(sino,              # [4,nty, ntz, nv, nu]  (iu contiguous)
    imgY,imgX,         
    p_bnd_arr, o_bnd_arr, z_bnd_arr,
    u_bnd, v_bnd,source_o,src_p_arr,src_z_arr,DSD):

    no, nz, nP = imgY.shape
    ns, nsrc_p, nsrc_z, nv, nu = sino.shape

    v0 = v_bnd[0]

    inv_dv = np.float32(1.0) / (v_bnd[1] - v_bnd[0])

    tmp_u = np.zeros((4,nu), dtype=np.float32)

    # OUTERMOST LOOP: orthagonal translations
    for io in range(no):

        denom = source_o - o_bnd_arr[io]
        if denom <= 0.0:
            continue  # avoid singularity / invalid region

        M = DSD/denom

        # Precompute projected y boundaries once per x-slice
        proj_p_bnd = M*p_bnd_arr

        # Precompute projected z boundaries once per x-slice
        proj_z_bnd = M*z_bnd_arr

        # Loop over z-slices (volume)
        for iz in range(nz):

            # Load contiguous y-column once (stays hot in cache)
            colX = imgX[io, iz, :]   # contiguous in ix
            colY = imgY[io, iz, :]   # contiguous in iy
            colXF = imgX[no-1-io, iz, :]   # contiguous in ix
            colYF = imgY[no-1-io, iz, :]   # contiguous in iy

            # Precompute projected z voxel boundaries
            z_vox_l = proj_z_bnd[iz]
            z_vox_r = proj_z_bnd[iz + 1]

            # Loop over parallel translations
            for i_sp in range(nsrc_p):
                src_p = src_p_arr[i_sp]

                # Reset u buffer
                tmp_u[:] = np.float32(0.0)

                # Monotonic overlap sweep in y
                ip = 0
                iu = 0

                p_l = proj_p_bnd[0] - M * src_p
                p_r = proj_p_bnd[1] - M * src_p

                u_l = u_bnd[0]
                u_r = u_bnd[1]

                while True:
                    overlap_l = max(p_l, u_l)
                    overlap_r = min(p_r, u_r)

                    overlap_u = overlap_r - overlap_l

                    if overlap_u>0:
                        tmp_u[0,iu] += overlap_u* colY[ip]
                        tmp_u[1,iu] += overlap_u* colX[nP - 1 - ip] 
                        tmp_u[2,iu] += overlap_u*colYF[nP - 1 - ip]
                        tmp_u[3,iu] += overlap_u*colXF[ip]

                    # Advance the interval that ends first
                    if p_r <= u_r:
                        ip += 1
                        if ip == nP:
                            break
                        p_l = p_r
                        p_r = proj_p_bnd[ip + 1] - M * src_p
                    else:
                        iu += 1
                        if iu == nu:
                            break
                        u_l = u_r
                        u_r = u_bnd[iu + 1]

                # Now reuse tmp_u across all z translations
                for i_sz in range(nsrc_z):
                    src_z = src_z_arr[i_sz]

                    vz_l = z_vox_l - M * src_z
                    vz_r = z_vox_r - M * src_z

                    iv_min = int((vz_l - v0) * inv_dv)
                    iv_max = int((vz_r - v0) * inv_dv) + 1

                    if iv_max <= 0 or iv_min >= nv:
                        continue

                    iv_min = max(0, iv_min)
                    iv_max = min(nv, iv_max)

                    for iv in range(iv_min, iv_max):
                        overlap_l = max(vz_l, v_bnd[iv])
                        overlap_r = min(vz_r, v_bnd[iv + 1])

                        overlap_v = overlap_r - overlap_l
                        if overlap_v>0:
                            # contiguous write in iu
                            sino[0,i_sp,i_sz,iv,:] += overlap_v*tmp_u[0,:] 
                            sino[1,i_sp,i_sz,iv,:] += overlap_v*tmp_u[1,:]
                            sino[2,i_sp,i_sz,iv,:] += overlap_v*tmp_u[2,:]
                            sino[3,i_sp,i_sz,iv,:] += overlap_v*tmp_u[3,:]
                            
"""