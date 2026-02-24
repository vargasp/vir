#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 21:38:08 2026

@author: varga
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


# Small epsilon to avoid numerical problems when a ray lies exactly
# on a grid line or boundary.
eps = np.float32(1e-6)


@njit(fastmath=True,inline='always',cache=True)
def _img_bounds(nr,dr):
    #Returns the image boundaries and center of first voxel
    return -dr*np.float32(nr)/2, dr*np.float32(nr)/2, -dr*(np.float32(nr)/2 - 0.5)


@njit(fastmath=True,inline='always',cache=True)
def _intersect_bounding(r0, dr, adr, r_min, r_max):
    # Intersect ray with image bounding box
    #
    # We compute parametric entry/exit values for a single component of r
    #   r(tr_enter) enters the image
    #   r(tr_exit) exits the image    
    if adr > eps:
        tr_entry = (r_min - r0) / dr
        tr_exit = (r_max - r0) / dr
        return min(tr_entry, tr_exit), max(tr_entry, tr_exit)
    else:
        # Ray is (almost) vertical → no x-bound intersection
        return -np.inf, np.inf

@njit(fastmath=True,inline='always',cache=True)
def _calc_ir0(ray_r_org,ray_r_hat,t_entry,img_bnd_r_min,img_bnd_r_max,d_pix):
    
    r = ray_r_org + t_entry*ray_r_hat
    r = min(max(r, img_bnd_r_min + eps), img_bnd_r_max - eps)
    
    # Convert to voxel indices
    return int(np.floor((r - img_bnd_r_min) / d_pix))
                    

@njit(fastmath=True,inline='always',cache=True)
def _fp_step_init(r0, ir, dr, adr, r_min, d_pix):   
    # Determine voxel traversal direction and next grid plane
    if dr > 0:
        # Ray moves toward increasing voxel indices
        idir = 1

        # Next grid plane is the "right" face of the current voxel
        # Grid planes are located at: r = r_min + k * d_pix
        r_next = r_min + (ir + 1)*d_pix
    else:
        # Ray moves toward decreasing voxel indices
        idir = -1

        # Next grid plane is the "left" face of the current voxel
        r_next = r_min + ir*d_pix

    # Compute ray-parameter increments for grid crossings
    if adr > eps:
        # Distance in t between successive grid-plane crossings
        # along this axis
        tr_step = d_pix / adr

        # Ray parameter t at which the ray first hits the next grid plane
        # Solve: r_next = r0 + t_next * dr
        tr_next = (r_next - r0) / dr
    else:
        # Ray is (nearly) parallel to the grid planes on this axis,
        # so it will never cross a plane here
        tr_step = np.inf
        tr_next = np.inf

    return idir, tr_step, tr_next
                    

@njit(fastmath=True,inline='always',cache=True)
def _aw_fp_traverse_2d(img,t_entry,t_exit,tx_next,ty_next,tx_step,ty_step,
                       ix,iy,ix_dir,iy_dir,nx,ny):
    
    # Ensure first crossing occurs after entry point
    tx_next = max(tx_next, t_entry)
    ty_next = max(ty_next, t_entry)

    # Traverse the grid voxel-by-voxel
    # At each step:
    #   - Choose the nearest boundary crossing
    #   - Accumulate voxel value × segment length
    #   - Advance to the next voxel
    t = t_entry
    acc = np.float32(0.0)

    while t <= t_exit:

        # Safety check (should rarely trigger)
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            break

        if tx_next <= ty_next:
            # Cross a vertical boundary first
            t_next = min(tx_next, t_exit)
            acc += img[ix, iy] * (t_next - t)
            t = t_next
            tx_next += tx_step
            ix += ix_dir
        else:
            # Cross a horizontal boundary first
            t_next = min(ty_next, t_exit)
            acc += img[ix, iy] * (t_next - t) 
            t = t_next
            ty_next += ty_step
            iy += iy_dir

    return acc

    
@njit(fastmath=True,inline='always',cache=True)
def _aw_bp_traverse_2d(img,s_val,t_entry,t_exit,tx_next,ty_next,
                       tx_step,ty_step,ix,iy,ix_dir,iy_dir,nx,ny):

    # Single ray: distribute sinogram to grid voxels 
    tx_next = max(tx_next, t_entry)
    ty_next = max(ty_next, t_entry)

    t = t_entry
    while t < t_exit:
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            break
            
        if tx_next <= ty_next:
            t_next = min(tx_next, t_exit)
            img[ix, iy] += s_val * (t_next - t)
            t = t_next
            tx_next += tx_step
            ix += ix_dir
        else:
            t_next = min(ty_next, t_exit)
            img[ix, iy] += s_val * (t_next - t)
            t = t_next
            ty_next += ty_step
            iy += iy_dir


@njit(fastmath=True,inline='always',cache=True)
def _aw_fp_traverse_3d(img,t_entry,t_exit,tx_next,ty_next,tz_next,
                       tx_step,ty_step,tz_step,
                       ix,iy,iz,ix_dir,iy_dir,iz_dir,nx,ny,nz):
    
    t = t_entry
    acc = np.float32(0.0)

    while t < t_exit:

        # Safety check (should rarely trigger)
        if ix<0 or iy<0 or iz<0 or ix>=nx or iy>=ny or iz>=nz:
            break

        if tx_next <= ty_next and tx_next <= tz_next:
            t_next = min(tx_next, t_exit)
            acc += img[ix,iy,iz]*(t_next - t)
            t = t_next
            tx_next += tx_step
            ix += ix_dir

        elif ty_next <= tz_next:
            t_next = min(ty_next, t_exit)
            acc += img[ix,iy,iz]*(t_next - t)
            t = t_next
            ty_next += ty_step
            iy += iy_dir

        else:
            t_next = min(tz_next, t_exit)
            acc += img[ix,iy,iz]*(t_next - t)
            t = t_next
            tz_next += tz_step
            iz += iz_dir

    return acc

@njit(fastmath=True,inline='always',cache=True)
def _aw_bp_traverse_3d(img,s_val,t_entry,t_exit,tx_next,ty_next,tz_next,
                       tx_step,ty_step,tz_step,
                       ix,iy,iz,ix_dir,iy_dir,iz_dir,nx,ny,nz):
    
    t = t_entry

    while t < t_exit:

        # Safety check (should rarely trigger)
        if ix<0 or iy<0 or iz<0 or ix>=nx or iy>=ny or iz>=nz:
            break

        if tx_next <= ty_next and tx_next <= tz_next:
            t_next = min(tx_next, t_exit)
            img[ix,iy,iz] += s_val*(t_next - t)
            t = t_next
            tx_next += tx_step
            ix += ix_dir

        elif ty_next <= tz_next:
            t_next = min(ty_next, t_exit)
            img[ix,iy,iz] += s_val*(t_next - t)
            t = t_next
            ty_next += ty_step
            iy += iy_dir

        else:
            t_next = min(tz_next, t_exit)
            img[ix,iy,iz] += s_val*(t_next - t)
            t = t_next
            tz_next += tz_step
            iz += iz_dir


@njit(fastmath=True,cache=True)
def _joseph_fp_2d(img,t_entry,t_exit,t_step,ray_x_org,ray_y_org,
                  ray_x_hat,ray_y_hat,x0,y0,d_pix):

    nx, ny = img.shape

    acc = 0.0 
    t = t_entry
    while t <= t_exit:
        # Current position along ray
        x = ray_x_org + t*ray_x_hat 
        y = ray_y_org + t*ray_y_hat
        
        # Convert to pixel index
        fx = (x - x0) / d_pix
        fy = (y - y0) / d_pix

        # Bilinear interpolation
        fx = min(max(fx, 0.0), nx - 2.0)
        fy = min(max(fy, 0.0), ny - 2.0)

        ix = int(fx)
        iy = int(fy)

        dx = fx - ix
        dy = fy - iy

        v00 = img[ix  , iy  ]
        v01 = img[ix  , iy+1]
        v10 = img[ix+1, iy  ]
        v11 = img[ix+1, iy+1]

        val = (v00*(1-dx)*(1-dy) +
               v10*   dx *(1-dy) +
               v01*(1-dx)*   dy  +
               v11*   dx *   dy)
        
        acc += val*t_step
        t += t_step

    return acc


@njit(fastmath=True,cache=True)
def _joseph_fp_3d(img,t_entry,t_exit,t_step,ray_x_org,ray_y_org,ray_z_org,
                  ray_x_hat,ray_y_hat,ray_z_hat,x0,y0,z0,d_pix):
    
    nx, ny, nz = img.shape

    acc = 0.0 
    t = t_entry
    while t <= t_exit:
        x = ray_x_org + t*ray_x_hat
        y = ray_y_org + t*ray_y_hat
        z = ray_z_org + t*ray_z_hat

        fx = (x - x0) / d_pix
        fy = (y - y0) / d_pix
        fz = (z - z0) / d_pix

        """
        if fx < 0 or fy < 0 or fz < 0 or \
           fx >= nx-1 or fy >= ny-1 or fz >= nz-1:
            t += t_step
            continue
        """
        
        ix = int(fx)
        iy = int(fy)
        iz = int(fz)
        
        if 0 <= ix < nx-1 and 0 <= iy < ny-1 and 0 <= iz < nz-1:
            dx = fx - ix
            dy = fy - iy
            dz = fz - iz
    
            # Trilinear interpolation
            c000 = img[ix  , iy  , iz  ]
            c100 = img[ix+1, iy  , iz  ]
            c010 = img[ix  , iy+1, iz  ]
            c110 = img[ix+1, iy+1, iz  ]
            c001 = img[ix  , iy  , iz+1]
            c101 = img[ix+1, iy  , iz+1]
            c011 = img[ix  , iy+1, iz+1]
            c111 = img[ix+1, iy+1, iz+1]
    
            # x interpolation
            c00 = c000 + dx * (c100 - c000)
            c10 = c010 + dx * (c110 - c010)
            c01 = c001 + dx * (c101 - c001)
            c11 = c011 + dx * (c111 - c011)
            
            # y interpolation
            c0 = c00 + dy * (c10 - c00)
            c1 = c01 + dy * (c11 - c01)
            
            # z interpolation
            val = c0 + dz * (c1 - c0)
            
            acc += val*t_step
            
        t += t_step

    return acc


@njit(fastmath=True,cache=True)
def _joseph_bp_2d(img, d_pix, s_val, ray_x_hat, ray_y_hat, ray_x_org, ray_y_org, x0, y0, t_enter, t_exit, step):
    nx, ny = img.shape

    t = t_enter
    while t <= t_exit:
        x = ray_x_org + t*ray_x_hat
        y = ray_y_org + t*ray_y_hat

        fx = (x - x0) / d_pix
        fy = (y - y0) / d_pix

        #ix = min(max(ix, 0.0), nx - 2.0)
        #iy = min(max(iy, 0.0), ny - 2.0)
        ix = int(np.floor(fx))
        iy = int(np.floor(fy))

        dx = fx - ix
        dy = fy - iy

        # Bilinear splatting
        if ix>=0 and iy>=0 and ix<nx and iy<ny:
            img[ix, iy]     += s_val*step*(1-dx)*(1-dy) 

        if ix+1>=0 and iy>=0 and ix+1<nx and iy<ny:
            img[ix+1, iy]   += s_val*step*   dx *(1-dy)

        if ix>=0 and iy+1>=0 and ix<nx and iy+1<ny:
            img[ix, iy+1]   += s_val*step*(1-dx)*   dy

        if ix+1>=0 and iy+1>=0 and ix+1<nx and iy+1<ny:
            img[ix+1, iy+1] += s_val*step*   dx *   dy

        t += step


@njit(fastmath=True,cache=True)
def _joseph_bp_3d(img,s_val,t_entry,t_exit,step,ray_x_org,ray_y_org,ray_z_org,
                  ray_x_hat,ray_y_hat,ray_z_hat,x0,y0,z0,d_pix):
    
    nx, ny, nz = img.shape


    t = t_entry
    while t <= t_exit:
        x = ray_x_org + t*ray_x_hat
        y = ray_y_org + t*ray_y_hat
        z = ray_z_org + t*ray_z_hat

        fx = (x - x0) / d_pix
        fy = (y - y0) / d_pix
        fz = (z - z0) / d_pix

        ix = int(np.floor(fx))
        iy = int(np.floor(fy))
        iz = int(np.floor(fz))
        
        dx = fx - ix
        dy = fy - iy
        dz = fz - iz

        # Trilinear splatting

        if ix>=0 and iy>=0 and iz>=0 and ix<nx and iy<ny and iz<nz:
            img[ix,iy,iz] += s_val*step*(1-dx)*(1-dy)*(1-dz) 


        if ix+1>=0 and iy>=0 and iz>=0 and ix+1<nx and iy<ny and iz<nz:
            img[ix+1,iy,iz] += s_val*step*(dx)*(1-dy)*(1-dz) 

        if ix>=0 and iy+1>=0 and iz>=0 and ix<nx and iy+1<ny and iz<nz:
            img[ix,iy+1,iz] += s_val*step*(1-dx)*(dy)*(1-dz) 

        if ix>=0 and iy>=0 and iz+1>=0 and ix<nx and iy<ny and iz+1<nz:
            img[ix,iy,iz+1] += s_val*step*(1-dx)*(1-dy)*(dz) 


        if ix>=0 and iy+1>=0 and iz+1>=0 and ix<nx and iy+1<ny and iz+1<nz:
            img[ix,iy+1,iz+1] += s_val*step*(1-dx)*(dy)*(dz) 

        if ix+1>=0 and iy>=0 and iz+1>=0 and ix+1<nx and iy<ny and iz+1<nz:
            img[ix+1,iy,iz+1] += s_val*step*(dx)*(1-dy)*(dz) 

        if ix+1>=0 and iy+1>=0 and iz>=0 and ix+1<nx and iy+1<ny and iz<nz:
            img[ix+1,iy+1,iz] += s_val*step*(dx)*(dy)*(1-dz) 

        if ix+1>=0 and iy+1>=0 and iz+1>=0 and ix+1<nx and iy+1<ny and iz+1<nz:
            img[ix+1,iy+1,iz+1] += s_val*step*(dx)*(dy)*(dz) 

        t += step



                    
def aw_fp_par_2d(img,ang_arr,nu,du=1.0,su=0.0,d_pix=1.0,joseph=False):
    
    img,ang_arr,du,su,d_pix = \
        as_float32(img,ang_arr,du,su,d_pix)
    
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    return aw_p_par_2d(img,sino,ang_arr,du=du,su=su,
                        d_pix=d_pix,joseph=joseph,bp=False)


def aw_bp_par_2d(sino,ang_arr,img_shape,du=1.0,su=0.0,d_pix=1.0,joseph=False):
    
    sino,ang_arr,du,su,d_pix = \
        as_float32(sino,ang_arr,du,su,d_pix)

    img = np.zeros(img_shape, dtype=np.float32)

    return aw_p_par_2d(img,sino,ang_arr,du=du,su=su,
                        d_pix=d_pix,joseph=joseph,bp=True)


@njit(fastmath=True,cache=True)
def aw_p_par_2d(img,sino,ang_arr,du,su,d_pix,joseph,bp):

    nx, ny = img.shape
    na, nu = sino.shape

    # Define image bounds in world coordinates
    img_bnd_x_min, img_bnd_x_max, x0 = _img_bounds(nx,d_pix)
    img_bnd_y_min, img_bnd_y_max, y0 = _img_bounds(ny,d_pix)
    step = np.float32(0.5)

    u_arr = du*(np.arange(nu,dtype=np.float32) - np.float32(nu/2 - 0.5) + su)
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    for ia, (cos_ang, sin_ang) in enumerate(zip(cos_ang_arr, sin_ang_arr)):
        
        det_u_orn = (-sin_ang, cos_ang)
        
        ray_x_hat = cos_ang
        ray_y_hat = sin_ang 
        
        rx_abs = abs(ray_x_hat)
        ry_abs = abs(ray_y_hat)

        for iu, u in enumerate(u_arr):
            ray_x_org = u*det_u_orn[0]
            ray_y_org = u*det_u_orn[1]

            tx_min, tx_max = _intersect_bounding(ray_x_org, ray_x_hat, rx_abs, img_bnd_x_min, img_bnd_x_max)
            ty_min, ty_max = _intersect_bounding(ray_y_org, ray_y_hat, ry_abs, img_bnd_y_min, img_bnd_y_max)
  
            t_entry = max(tx_min, ty_min)
            t_exit  = min(tx_max, ty_max)

            if t_exit <= t_entry:
                continue


            if joseph:
                if bp:
                    _joseph_bp_2d(img, d_pix, sino[ia,iu], cos_ang, sin_ang, ray_x_org, ray_y_org,
                           x0, y0, t_entry, t_exit, step)
                else:
                    sino[ia, iu] = _joseph_fp_2d(img,t_entry,t_exit,step,
                                             ray_x_org,ray_y_org,
                                             ray_x_hat,ray_y_hat,x0,y0,d_pix)
                
            else:
                
                ix_entry = _calc_ir0(ray_x_org,ray_x_hat,t_entry,img_bnd_x_min,img_bnd_x_max,d_pix)
                iy_entry = _calc_ir0(ray_y_org,ray_y_hat,t_entry,img_bnd_y_min,img_bnd_y_max,d_pix)
                
                ix_dir, tx_step, tx_next = _fp_step_init(ray_x_org, ix_entry, ray_x_hat, rx_abs, img_bnd_x_min, d_pix)
                iy_dir, ty_step, ty_next = _fp_step_init(ray_y_org, iy_entry, ray_y_hat, ry_abs, img_bnd_y_min, d_pix)

                if bp:
                    _aw_bp_traverse_2d(img, sino[ia, iu],t_entry,t_exit,
                            tx_next,ty_next,tx_step,ty_step, 
                            ix_entry,iy_entry,ix_dir,iy_dir,nx,ny)
                else:
                # Traverse the grid voxel-by-voxel
                    sino[ia, iu] = _aw_fp_traverse_2d(img,t_entry,t_exit,
                                     tx_next, ty_next, tx_step, ty_step,
                                     ix_entry,iy_entry,ix_dir,iy_dir,nx,ny,)
                
    if bp:
        return img / na / d_pix / d_pix * du
    else:
        return sino


def aw_fp_fan_2d(img,ang_arr,nu,DSO,DSD,du=1.0,su=0.0,d_pix=1.0,joseph=False):
    
    img,ang_arr,du,su,d_pix = \
        as_float32(img,ang_arr,du,su,d_pix)
    
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    return aw_p_fan_2d(img,sino,ang_arr,DSO,DSD,du=du,su=su,
                        d_pix=d_pix,joseph=joseph,bp=False)



def aw_bp_fan_2d(sino,ang_arr,img_shape,DSO,DSD,du=1.0,su=0.0,d_pix=1.0,joseph=False):
    
    sino,ang_arr,du,su,d_pix = \
        as_float32(sino,ang_arr,du,su,d_pix)

    img = np.zeros(img_shape, dtype=np.float32)

    return aw_p_fan_2d(img,sino,ang_arr,DSO,DSD,du=du,su=su,
                        d_pix=d_pix,joseph=joseph,bp=True)

@njit(fastmath=True,cache=True)
def aw_p_fan_2d(img,sino,ang_arr,DSO,DSD,du,su,d_pix,joseph,bp):

    nx, ny = img.shape
    na, nu = sino.shape

    img_bnd_x_min, img_bnd_x_max, x0 = _img_bounds(nx,d_pix)
    img_bnd_y_min, img_bnd_y_max, y0 = _img_bounds(ny,d_pix)
    step = np.float32(0.5)


    u_arr = du*(np.arange(nu,dtype=np.float32) - np.float32(nu/2 - 0.5) + su)
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    for ia, (cos_ang, sin_ang) in enumerate(zip(cos_ang_arr, sin_ang_arr)):
        
        # Ray origin (located at the source)
        ray_x_org = DSO * cos_ang
        ray_y_org = DSO * sin_ang
        
        
        # Detector reference point
        det_x_org = -(DSD - DSO)*cos_ang
        det_y_org = -(DSD - DSO)*sin_ang
        
        # Ray direction (parallel abd orthoganal to detector)
        det_u_orn = (-sin_ang, cos_ang)

        for iu, u in enumerate(u_arr):
            det_x = det_x_org + u*det_u_orn[0]
            det_y = det_y_org + u*det_u_orn[1]

            # Ray from source to detector
            ray_x_vec = det_x - ray_x_org
            ray_y_vec = det_y - ray_y_org

            #Ray directions in unit vector components
            ray_mag = np.sqrt(ray_x_vec**2 + ray_y_vec**2)
            ray_x_hat = ray_x_vec/ray_mag
            ray_y_hat = ray_y_vec/ray_mag
            
            tx_entry, tx_exit = _intersect_bounding(ray_x_org, ray_x_hat, abs(ray_x_hat), img_bnd_x_min, img_bnd_x_max)
            ty_entry, ty_exit = _intersect_bounding(ray_y_org, ray_y_hat, abs(ray_y_hat), img_bnd_y_min, img_bnd_y_max)

            t_entry = max(tx_entry, ty_entry)
            t_exit  = min(tx_exit, ty_exit)

            if t_exit <= t_entry:
                continue

            if joseph:
                if bp:
                    _joseph_bp_2d(img, d_pix, sino[ia,iu], ray_x_hat, ray_y_hat, ray_x_org, ray_y_org,
                           x0, y0, t_entry, t_exit, step)
                else:
                    sino[ia, iu] = _joseph_fp_2d(img,t_entry,t_exit,step,
                                             ray_x_org,ray_y_org,
                                             ray_x_hat,ray_y_hat,x0,y0,d_pix)

            else:
                ix_entry = _calc_ir0(ray_x_org,ray_x_hat,t_entry,img_bnd_x_min,img_bnd_x_max,d_pix)
                iy_entry = _calc_ir0(ray_y_org,ray_y_hat,t_entry,img_bnd_y_min,img_bnd_y_max,d_pix)
    
                # Amanatides–Woo stepping initialization
                ix_dir, tx_step, tx_next = _fp_step_init(ray_x_org, ix_entry, ray_x_hat, abs(ray_x_hat), img_bnd_x_min, d_pix)
                iy_dir, ty_step, ty_next = _fp_step_init(ray_y_org, iy_entry, ray_y_hat, abs(ray_y_hat), img_bnd_y_min, d_pix)


                if bp:    
      
                    _aw_bp_traverse_2d(img, sino[ia, iu],t_entry,t_exit,
                                       tx_next,ty_next,tx_step,ty_step,
                                       ix_entry,iy_entry,ix_dir,iy_dir,nx,ny)
                else:
                    sino[ia, iu] = _aw_fp_traverse_2d(img,t_entry,t_exit,
                                     tx_next,ty_next,tx_step,ty_step,
                                     ix_entry,iy_entry,ix_dir,iy_dir,nx,ny)
                    

    if bp:
        return img / na / d_pix / d_pix * du
    else:
        return sino


def aw_fp_cone_3d(img,ang_arr,nu,nv,DSO,DSD,
                  du=1.0,dv=1.0,su=0.0,sv=0.0,d_pix=1.0,joseph=False):
    
    img,ang_arr,DSO,DSD,du,dv,su,sv,d_pix = \
        as_float32(img,ang_arr,DSO,DSD,du,dv,su,sv,d_pix)
    
    sino = np.zeros((ang_arr.size, nu, nv), dtype=np.float32)

    return aw_p_cone_3d(img,sino,ang_arr,DSO,DSD,du=du,dv=dv,su=su,sv=sv,
                        d_pix=d_pix,joseph=joseph,bp=False)



def aw_bp_cone_3d(sino,ang_arr,img_shape,DSO,DSD,
                  du=1.0,dv=1.0,su=0.0,sv=0.0,d_pix=1.0,joseph=False):
    
    sino,ang_arr,DSO,DSD,du,dv,su,sv,d_pix = \
        as_float32(sino,ang_arr,DSO,DSD,du,dv,su,sv,d_pix)

    img = np.zeros(img_shape, dtype=np.float32)

    return aw_p_cone_3d(img,sino,ang_arr,DSO,DSD,du=du,dv=dv,su=su,sv=sv,
                        d_pix=d_pix,joseph=joseph,bp=True)


@njit(fastmath=True,cache=True)
def aw_p_cone_3d(img,sino,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,joseph,bp):

    nx, ny, nz = img.shape
    na, nu, nv = sino.shape
    
    img_bnd_x_min, img_bnd_x_max, x0 = _img_bounds(nx,d_pix)
    img_bnd_y_min, img_bnd_y_max, y0 = _img_bounds(ny,d_pix)
    img_bnd_z_min, img_bnd_z_max, z0 = _img_bounds(nz,d_pix)
    step = np.float32(0.5)

    u_arr = du*(np.arange(nu,dtype=np.float32) - np.float32(nu/2 - 0.5) + su)
    v_arr = dv*(np.arange(nv,dtype=np.float32) - np.float32(nu/2 - 0.5) + sv)

    # Precompute ray direction for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    # Main loops: angles → detectors → voxel traversal
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):

        # Ray origin (located at the source)
        ray_x_org = DSO*cos_ang
        ray_y_org = DSO*sin_ang
        ray_z_org = np.float32(0.0)
        
        # Detector origin point
        det_x_org = -(DSD - DSO)*cos_ang
        det_y_org = -(DSD - DSO)*sin_ang
        det_z_org = np.float32(0.0)

        # Detector basis orientation (unit vectors)
        #det_u_orn = (-sin_ang, cos_ang, 0)
        #det_v_orn = (0, 0, 1)

        #Detector positions
        #det_x = det_x_org + u*det_u_orn[0] + v*det_v_orn[0]
        det_x_arr = det_x_org + u_arr * -sin_ang
        det_y_arr = det_y_org + u_arr * cos_ang

        for iv, v in enumerate(v_arr):
            #Detector position z
            det_z = det_z_org + v

            for iu, u in enumerate(u_arr):
                #Detector position x,y
                det_x = det_x_arr[iu]
                det_y = det_y_arr[iu]

                #Ray vector 
                ray_x_vec = det_x - ray_x_org
                ray_y_vec = det_y - ray_y_org
                ray_z_vec = det_z - ray_z_org


                ray_mag = np.sqrt(ray_x_vec**2 + ray_y_vec**2 + ray_z_vec**2)
                ray_x_hat = ray_x_vec/ray_mag
                ray_y_hat = ray_y_vec/ray_mag
                ray_z_hat = ray_z_vec/ray_mag

                tx_entry,tx_exit = _intersect_bounding(ray_x_org, ray_x_hat, abs(ray_x_hat), img_bnd_x_min, img_bnd_x_max)
                ty_entry,ty_exit = _intersect_bounding(ray_y_org, ray_y_hat, abs(ray_y_hat), img_bnd_y_min, img_bnd_y_max)
                tz_entry,tz_exit = _intersect_bounding(ray_z_org, ray_z_hat, abs(ray_z_hat), img_bnd_z_min, img_bnd_z_max)

                t_entry = max(tx_entry, ty_entry, tz_entry)
                t_exit  = min(tx_exit, ty_exit, tz_exit)


                if t_exit <= t_entry:
                    continue

                if joseph:

                    if bp:
                        _joseph_bp_3d(img,sino[ia,iu,iv],t_entry,t_exit,step,
                                                       ray_x_org,ray_y_org,ray_z_org,
                                                       ray_x_hat,ray_y_hat,ray_z_hat,
                                                       x0,y0,z0,d_pix)
                    else:
                        sino[ia,iu,iv] = _joseph_fp_3d(img,t_entry,t_exit,step,
                                                   ray_x_org,ray_y_org,ray_z_org,
                                                   ray_x_hat,ray_y_hat,ray_z_hat,
                                                   x0,y0,z0,d_pix)
                else:
                    ix_entry = _calc_ir0(ray_x_org,ray_x_hat,t_entry,
                                         img_bnd_x_min,img_bnd_x_max,d_pix)
                    iy_entry = _calc_ir0(ray_y_org,ray_y_hat,t_entry,
                                         img_bnd_y_min,img_bnd_y_max,d_pix)
                    iz_entry = _calc_ir0(ray_z_org,ray_z_hat,t_entry,
                                         img_bnd_z_min,img_bnd_z_max,d_pix)

                    ix_dir,tx_step,tx_next = _fp_step_init(ray_x_org,ix_entry, ray_x_hat, abs(ray_x_hat), img_bnd_x_min, d_pix)
                    iy_dir,ty_step,ty_next = _fp_step_init(ray_y_org,iy_entry, ray_y_hat, abs(ray_y_hat), img_bnd_y_min, d_pix)
                    iz_dir,tz_step,tz_next = _fp_step_init(ray_z_org,iz_entry, ray_z_hat, abs(ray_z_hat), img_bnd_z_min, d_pix)

                    if bp:
                        _aw_bp_traverse_3d(img,sino[ia,iu,iv],t_entry,t_exit,
                            tx_next,ty_next,tz_next,tx_step, ty_step, tz_step,
                            ix_entry, iy_entry, iz_entry,
                            ix_dir,iy_dir,iz_dir,nx,ny,nz)

                    else:
                        sino[ia,iu,iv] = _aw_fp_traverse_3d(img,t_entry, t_exit,
                            tx_next,ty_next,tz_next,tx_step, ty_step, tz_step,
                            ix_entry, iy_entry, iz_entry,
                            ix_dir,iy_dir,iz_dir,nx,ny,nz)
                                        
                        ray_scale = DSD/np.sqrt(DSD**2 + u**2 + v**2)
                        sino[ia, iu, iv]= sino[ia, iu, iv]*ray_scale
                    
    if bp:
        return img/ang_arr.size*du*dv
    else:
        return sino
    


def aw_p_square(img,nu,nv,ns_p,ns_z,DSO,DSD,
                  du=1.0,dv=1.0,ds_p=1.0,ds_z=1.0, 
                  su=0.0,sv=0.0,d_pix=1.0,joseph=False):
    
    sino = np.zeros([4,ns_p,ns_z,nu,nv], np.float32)
   # img,sino,DSO,DSD,du,dv,dsrc_p,dsrc_z,d_pix = \
   #     as_float32(img,sino,DSO,DSD,du,dv,dsrc_p,dsrc_z,d_pix)
    

    return aw_p_square_num(img,sino,DSO,DSD,du,dv,ds_p,ds_z,su,sv,d_pix)


@njit(inline='always')
def setup_axis(r_not_parallel,ray_r_hat,img_bnd_r_min, img_bnd_r_max,src_r0,dsrc,d_pix,nr):

    if r_not_parallel:
        inv_ray_r_hat = 1.0 / ray_r_hat

        tr1 = img_bnd_r_min * inv_ray_r_hat
        tr2 = img_bnd_r_max * inv_ray_r_hat

        tr_entry_base = min(tr1, tr2) - src_r0*inv_ray_r_hat
        tr_exit_base  = max(tr1, tr2) - src_r0*inv_ray_r_hat

        tr_pix_step = d_pix / np.abs(ray_r_hat)
        tr_src_step = -dsrc * inv_ray_r_hat

        ir_plane = 0 if ray_r_hat > 0 else nr - 1

        return inv_ray_r_hat, tr_entry_base, tr_exit_base, tr_pix_step, tr_src_step, ir_plane

    else:
        return 0.0, -np.inf, np.inf, np.inf, 0.0, 0



@njit(fastmath=True, cache=True)
def aw_p_square_num(img,sino,DSO,DSD,du,dv,dsrc_p,dsrc_z,su,sv,d_pix):

    nx, ny, nz = img.shape
    ns, nsrc_p, nsrc_z, nu, nv = sino.shape

    # Image bounds
    img_bnd_x_min, img_bnd_x_max, _ = _img_bounds(nx, d_pix)
    img_bnd_y_min, img_bnd_y_max, _ = _img_bounds(ny, d_pix)
    img_bnd_z_min, img_bnd_z_max, _ = _img_bounds(nz, d_pix)

    # Detector coordinates
    u_arr = pf.censpace(du,nu,su)
    v_arr = pf.censpace(dv,nv,sv)

    # Source coordinates
    src_p_arr = pf.censpace(dsrc_p,nsrc_p)
    src_z_arr = pf.censpace(dsrc_z,nsrc_z)


    inv_dpix = 1.0 / d_pix
                
    side_x_arr = ( 0.0, -1.0,  0.0, 1.0)
    side_y_arr = ( 1.0,  0.0, -1.0, 0.0)

    norm_x_arr = (-1.0,  0.0,  1.0, 0.0)
    norm_y_arr = ( 0.0, -1.0,  0.0, 1.0)
    
    
    for iside in range(ns):

        side_x = side_x_arr[iside]
        side_y = side_y_arr[iside]
        
        norm_x = norm_x_arr[iside]
        norm_y = norm_y_arr[iside]
        
        base_x = -DSO * norm_x
        base_y = -DSO * norm_y
        
        dsrc_x = dsrc_p * side_x
        dsrc_y = dsrc_p * side_y

        # Precompute translated source arrays
        src_x_arr = base_x + src_p_arr*side_x
        src_y_arr = base_y + src_p_arr*side_y
        
        
                
        for iu in range(nu):
            u_det = u_arr[iu]

            # Ray components independent of source translation
            ray_x_vec = DSD*norm_x + u_det*side_x
            ray_y_vec = DSD*norm_y + u_det*side_y

            for iv in range(nv):
                ray_z_vec = v_arr[iv]

                r_mag_inv = 1.0/np.sqrt(ray_x_vec**2 + ray_y_vec**2 + ray_z_vec**2)

                ray_x_hat = ray_x_vec*r_mag_inv
                ray_y_hat = ray_y_vec*r_mag_inv
                ray_z_hat = ray_z_vec*r_mag_inv

                x_not_parallel = abs(ray_x_hat) > eps
                y_not_parallel = abs(ray_y_hat) > eps
                z_not_parallel = abs(ray_z_hat) > eps

                ix_dir = 1 if ray_x_hat > 0 else -1
                iy_dir = 1 if ray_y_hat > 0 else -1
                iz_dir = 1 if ray_z_hat > 0 else -1

                inv_rx,tx_entry_base,tx_exit_base,tx_step,dtx,ix_plane = \
                    setup_axis(x_not_parallel,ray_x_hat,img_bnd_x_min,img_bnd_x_max,src_x_arr[0],dsrc_x,d_pix,nx)

                inv_ry,ty_entry_base,ty_exit_base,ty_step,dty,iy_plane = \
                    setup_axis(y_not_parallel,ray_y_hat,img_bnd_y_min,img_bnd_y_max,src_y_arr[0],dsrc_y,d_pix,ny)
                
                inv_rz,tz_entry_base,tz_exit_base,tz_step,dtz,iz_plane = \
                    setup_axis(z_not_parallel,ray_z_hat,img_bnd_z_min,img_bnd_z_max,src_z_arr[0],dsrc_z,d_pix,nz)
                

                # Loop over translated sources (parallel)
                for i_sp in range(nsrc_p):
                    ray_x_org = src_x_arr[i_sp] - img_bnd_x_min
                    ray_y_org = src_y_arr[i_sp] - img_bnd_y_min

                    # Shift x and y intersections linearly
                    if x_not_parallel:
                        tx_entry = tx_entry_base + i_sp*dtx
                        tx_exit  = tx_exit_base  + i_sp*dtx

                    if y_not_parallel:
                        ty_entry = ty_entry_base + i_sp*dty
                        ty_exit  = ty_exit_base  + i_sp*dty


                    # Loop over translated sources (z)
                    for i_sz in range(nsrc_z):
                        ray_z_org = src_z_arr[i_sz] - img_bnd_z_min

                        # Shift z intersections linearly
                        if z_not_parallel:
                            tz_entry = tz_entry_base + i_sz*dtz
                            tz_exit  = tz_exit_base  + i_sz*dtz

                        # Determine entry parameter
                        t_entry = max(tx_entry, ty_entry, tz_entry)
                        t_exit  = min(tx_exit, ty_exit, tz_exit)
                        
                        if t_exit <= t_entry:
                            continue
                        
                        # Determine dominant axis
                        if tx_entry >= ty_entry and tx_entry >= tz_entry:
                            # Enter through x-plane
                            ix_entry = ix_plane
                        
                            ry_entry = ray_y_org + t_entry * ray_y_hat
                            rz_entry = ray_z_org + t_entry * ray_z_hat
                        
                            iy_entry = int(ry_entry*inv_dpix)
                            iz_entry = int(rz_entry*inv_dpix)
                        
                        elif ty_entry >= tz_entry:
                            # Enter through y-plane
                            iy_entry = iy_plane
                        
                            rx_entry = ray_x_org + t_entry * ray_x_hat
                            rz_entry = ray_z_org + t_entry * ray_z_hat
                        
                            ix_entry = int(rx_entry*inv_dpix)
                            iz_entry = int(rz_entry*inv_dpix)
                        
                        else:
                            # Enter through z-plane
                            iz_entry = iz_plane
                        
                            rx_entry = ray_x_org + t_entry * ray_x_hat
                            ry_entry = ray_y_org + t_entry * ray_y_hat
                        
                            ix_entry = int(rx_entry*inv_dpix)
                            iy_entry = int(ry_entry*inv_dpix)

                        ix_entry = max(0, min(ix_entry, nx - 1))
                        iy_entry = max(0, min(iy_entry, ny - 1))
                        iz_entry = max(0, min(iz_entry, nz - 1))

                        # Step init                       
                        if x_not_parallel:
                            r_next_x = (ix_entry + (ix_dir > 0)) * d_pix
                            tx_next = (r_next_x - ray_x_org) * inv_rx
                       
                        if y_not_parallel:
                            r_next_y = (iy_entry + (iy_dir > 0)) * d_pix
                            ty_next = (r_next_y - ray_y_org) * inv_ry
                                                    
                        if z_not_parallel:
                            r_next_z = (iz_entry + (iz_dir > 0)) * d_pix
                            tz_next = (r_next_z - ray_z_org) * inv_rz


                        # Traverse
                        val = _aw_fp_traverse_3d(
                            img,
                            t_entry, t_exit,
                            tx_next, ty_next, tz_next,
                            tx_step, ty_step, tz_step,
                            ix_entry, iy_entry, iz_entry,
                            ix_dir, iy_dir, iz_dir,
                            nx, ny, nz
                        )

                        sino[iside, i_sp, i_sz, iu, iv] = val * (DSD * r_mag_inv)

    return sino
