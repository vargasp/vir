#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 21:38:08 2026

@author: varga
"""


def _identity_decorator(func):
    return func

try:
    from numbas import njit
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
                  du=1.0,dv=1.0,su=0.0,sv=1.0,d_pix=1.0,joseph=False):
    
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
    



@njit(fastmath=True, cache=True)
def aw_p_square(img, sino,
                                           DSO, DSD,
                                           du, dv,
                                           d_pix):
    """
    Square trajectory forward projector.
    Loops in order: side → detector u → detector v → source translation p → source z
    """

    nx, ny, nz = img.shape
    ns, ns_p, ns_z, nu, nv = sino.shape

    # Image bounds
    img_bnd_x_min, img_bnd_x_max, x0 = _img_bounds(nx, d_pix)
    img_bnd_y_min, img_bnd_y_max, y0 = _img_bounds(ny, d_pix)
    img_bnd_z_min, img_bnd_z_max, z0 = _img_bounds(nz, d_pix)

    # Detector coordinates along u and v
    u0_arr = du * (np.arange(nu, dtype=np.float32) - (nu / 2 - 0.5))
    v_arr = dv * (np.arange(nv, dtype=np.float32) - (nv / 2 - 0.5))

    # Source translations along side and in z
    p_arr = np.linspace(-DSO, DSO, ns_p).astype(np.float32)
    sz_arr = np.linspace(img_bnd_z_min, img_bnd_z_max, ns_z).astype(np.float32)

    for iside in range(ns):
        # Side geometry: determine which axis the source moves along
        if iside == 0:       # Bottom side: X moves, Y fixed
            side_dir_x, side_dir_y = 1.0, 0.0
            norm_x, norm_y = 0.0, 1.0  # detector plane normal along Y
            base_x, base_y = 0.0, -DSO
        elif iside == 1:     # Right side: Y moves, X fixed
            side_dir_x, side_dir_y = 0.0, 1.0
            norm_x, norm_y = -1.0, 0.0
            base_x, base_y = DSO, 0.0
        elif iside == 2:     # Top side: X moves, Y fixed
            side_dir_x, side_dir_y = -1.0, 0.0
            norm_x, norm_y = 0.0, -1.0
            base_x, base_y = 0.0, DSO
        else:                # Left side: Y moves, X fixed
            side_dir_x, side_dir_y = 0.0, -1.0
            norm_x, norm_y = 1.0, 0.0
            base_x, base_y = -DSO, 0.0

        for iu in range(nu):
            u_det = u0_arr[iu]

            for iv in range(nv):
                v_det = v_arr[iv]

                for ip in range(ns_p):
                    p_shift = p_arr[ip]

                    # Source position along the side
                    src_x = base_x + p_shift * side_dir_x
                    src_y = base_y + p_shift * side_dir_y

                    for izs in range(ns_z):
                        src_z = sz_arr[izs]

                        # Detector position
                        det_x = src_x + DSD * norm_x + u_det * side_dir_x
                        det_y = src_y + DSD * norm_y + u_det * side_dir_y
                        det_z = src_z + v_det

                        # Ray vector
                        ray_x_vec = det_x - src_x
                        ray_y_vec = det_y - src_y
                        ray_z_vec = det_z - src_z
                        ray_mag = np.sqrt(ray_x_vec**2 + ray_y_vec**2 + ray_z_vec**2)

                        ray_x_hat = ray_x_vec / ray_mag
                        ray_y_hat = ray_y_vec / ray_mag
                        ray_z_hat = ray_z_vec / ray_mag

                        # Intersections with image bounds
                        tx_entry, tx_exit = _intersect_bounding(src_x, ray_x_hat, abs(ray_x_hat),
                                                                img_bnd_x_min, img_bnd_x_max)
                        ty_entry, ty_exit = _intersect_bounding(src_y, ray_y_hat, abs(ray_y_hat),
                                                                img_bnd_y_min, img_bnd_y_max)
                        tz_entry, tz_exit = _intersect_bounding(src_z, ray_z_hat, abs(ray_z_hat),
                                                                img_bnd_z_min, img_bnd_z_max)

                        t_entry = max(tx_entry, ty_entry, tz_entry)
                        t_exit  = min(tx_exit, ty_exit, tz_exit)

                        if t_exit <= t_entry:
                            continue

                        # Entry voxel indices
                        ix = _calc_ir0(src_x, ray_x_hat, t_entry, img_bnd_x_min, img_bnd_x_max, d_pix)
                        iy = _calc_ir0(src_y, ray_y_hat, t_entry, img_bnd_y_min, img_bnd_y_max, d_pix)
                        iz = _calc_ir0(src_z, ray_z_hat, t_entry, img_bnd_z_min, img_bnd_z_max, d_pix)

                        # Initialize voxel stepping
                        ix_dir, tx_step, tx_next = _fp_step_init(src_x, ix, ray_x_hat, abs(ray_x_hat),
                                                                 img_bnd_x_min, d_pix)
                        iy_dir, ty_step, ty_next = _fp_step_init(src_y, iy, ray_y_hat, abs(ray_y_hat),
                                                                 img_bnd_y_min, d_pix)
                        iz_dir, tz_step, tz_next = _fp_step_init(src_z, iz, ray_z_hat, abs(ray_z_hat),
                                                                 img_bnd_z_min, d_pix)

                        # Voxel traversal
                        val = _aw_fp_traverse_3d(
                            img,
                            t_entry, t_exit,
                            tx_next, ty_next, tz_next,
                            tx_step, ty_step, tz_step,
                            ix, iy, iz,
                            ix_dir, iy_dir, iz_dir,
                            nx, ny, nz
                        )

                        # Store in sinogram
                        sino[iside, ip, izs, iu, iv] = val * (DSD / ray_mag)

    return sino


@njit(fastmath=True, cache=True)
def aw_p_square2(img, sino, DSO, DSD, du, dv, d_pix):
    """
    Square trajectory forward projector.
    Loops in order: side → detector u → detector v → source translation p → source z
    Optimized: ray vectors precomputed per detector element (iu, iv) since source and detector move together.
    """

    nx, ny, nz = img.shape
    ns, ns_p, ns_z, nu, nv = sino.shape

    # Image bounds
    img_bnd_x_min, img_bnd_x_max, x0 = _img_bounds(nx, d_pix)
    img_bnd_y_min, img_bnd_y_max, y0 = _img_bounds(ny, d_pix)
    img_bnd_z_min, img_bnd_z_max, z0 = _img_bounds(nz, d_pix)

    # Detector coordinates along u and v
    u0_arr = du * (np.arange(nu, dtype=np.float32) - (nu / 2 - 0.5))
    v_arr = dv * (np.arange(nv, dtype=np.float32) - (nv / 2 - 0.5))

    # Source translations along side and in z
    p_arr = np.linspace(-DSO, DSO, ns_p).astype(np.float32)
    sz_arr = np.linspace(img_bnd_z_min, img_bnd_z_max, ns_z).astype(np.float32)

    for iside in range(ns):
        # Side geometry
        if iside == 0:       # Bottom side: X moves, Y fixed
            side_dir_x, side_dir_y = 1.0, 0.0
            norm_x, norm_y = 0.0, 1.0
            base_x, base_y = 0.0, -DSO
        elif iside == 1:     # Right side: Y moves, X fixed
            side_dir_x, side_dir_y = 0.0, 1.0
            norm_x, norm_y = -1.0, 0.0
            base_x, base_y = DSO, 0.0
        elif iside == 2:     # Top side: X moves, Y fixed
            side_dir_x, side_dir_y = -1.0, 0.0
            norm_x, norm_y = 0.0, -1.0
            base_x, base_y = 0.0, DSO
        else:                # Left side: Y moves, X fixed
            side_dir_x, side_dir_y = 0.0, -1.0
            norm_x, norm_y = 1.0, 0.0
            base_x, base_y = -DSO, 0.0

        src_x0 = base_x
        src_y0 = base_y
        src_z0 = 0.0

        # Now loop over detector and sources
        for iu in range(nu):
            u_det = u0_arr[iu]
            
            det_x0 = src_x0 + DSD * norm_x + u_det * side_dir_x
            det_y0 = base_y + DSD * norm_y + u_det * side_dir_y
            rx = det_x0 - src_x0
            ry = det_y0 - src_y0

            for iv in range(nv):
                v_det = v_arr[iv]

                det_z0 = src_z0 + v_det
                rz = det_z0 - src_z0

                rmag = np.sqrt(rx*rx + ry*ry + rz*rz)
                ray_x_hat = rx / rmag
                ray_y_hat = ry / rmag
                ray_z_hat = rz / rmag

                for ip in range(ns_p):
                    p_shift = p_arr[ip]

                    # Source moves along side
                    src_x = base_x + p_shift * side_dir_x
                    src_y = base_y + p_shift * side_dir_y

                    for izs in range(ns_z):
                        src_z = sz_arr[izs]


                        # Intersections with image bounds
                        tx_entry, tx_exit = _intersect_bounding(src_x, ray_x_hat, abs(ray_x_hat),
                                                                img_bnd_x_min, img_bnd_x_max)
                        ty_entry, ty_exit = _intersect_bounding(src_y, ray_y_hat, abs(ray_y_hat),
                                                                img_bnd_y_min, img_bnd_y_max)
                        tz_entry, tz_exit = _intersect_bounding(src_z, ray_z_hat, abs(ray_z_hat),
                                                                img_bnd_z_min, img_bnd_z_max)

                        t_entry = max(tx_entry, ty_entry, tz_entry)
                        t_exit  = min(tx_exit, ty_exit, tz_exit)

                        if t_exit <= t_entry:
                            continue

                        # Entry voxel indices
                        ix = _calc_ir0(src_x, ray_x_hat, t_entry, img_bnd_x_min, img_bnd_x_max, d_pix)
                        iy = _calc_ir0(src_y, ray_y_hat, t_entry, img_bnd_y_min, img_bnd_y_max, d_pix)
                        iz = _calc_ir0(src_z, ray_z_hat, t_entry, img_bnd_z_min, img_bnd_z_max, d_pix)

                        # Initialize voxel stepping
                        ix_dir, tx_step, tx_next = _fp_step_init(src_x, ix, ray_x_hat, abs(ray_x_hat),
                                                                 img_bnd_x_min, d_pix)
                        iy_dir, ty_step, ty_next = _fp_step_init(src_y, iy, ray_y_hat, abs(ray_y_hat),
                                                                 img_bnd_y_min, d_pix)
                        iz_dir, tz_step, tz_next = _fp_step_init(src_z, iz, ray_z_hat, abs(ray_z_hat),
                                                                 img_bnd_z_min, d_pix)

                        # Voxel traversal
                        val = _aw_fp_traverse_3d(
                            img,
                            t_entry, t_exit,
                            tx_next, ty_next, tz_next,
                            tx_step, ty_step, tz_step,
                            ix, iy, iz,
                            ix_dir, iy_dir, iz_dir,
                            nx, ny, nz
                        )

                        sino[iside, ip, izs, iu, iv] = val * (DSD / rmag)

    return sino


@njit(fastmath=True, cache=True)
def aw_p_square3(img, sino, DSO, DSD, du, dv, d_pix):

    nx, ny, nz = img.shape
    ns, ns_p, ns_z, nu, nv = sino.shape

    # Image bounds
    bx_min, bx_max, _ = _img_bounds(nx, d_pix)
    by_min, by_max, _ = _img_bounds(ny, d_pix)
    bz_min, bz_max, _ = _img_bounds(nz, d_pix)

    # Detector coordinates
    u_arr = du * (np.arange(nu, dtype=np.float32) - (nu / 2 - 0.5))
    v_arr = dv * (np.arange(nv, dtype=np.float32) - (nv / 2 - 0.5))

    # Source sampling
    p_arr  = np.linspace(-DSO, DSO, ns_p).astype(np.float32)
    sz_arr = np.linspace(bz_min, bz_max, ns_z).astype(np.float32)

    for iside in range(ns):

        # --- Side geometry ---
        if iside == 0:
            side_x, side_y = 1.0, 0.0
            norm_x, norm_y = 0.0, 1.0
            base_x, base_y = 0.0, -DSO
            src_x_arr = base_x + p_arr * side_x
            src_y_arr = base_y + p_arr * side_y
            
        elif iside == 1:
            side_x, side_y = 0.0, 1.0
            norm_x, norm_y = -1.0, 0.0
            base_x, base_y = DSO, 0.0
            src_x_arr = base_x + p_arr * side_x
            src_y_arr = base_y + p_arr * side_y

        elif iside == 2:
            side_x, side_y = -1.0, 0.0
            norm_x, norm_y = 0.0, -1.0
            base_x, base_y = 0.0, DSO
            src_x_arr = base_x + p_arr * side_x
            src_y_arr = base_y + p_arr * side_y

        else:
            side_x, side_y = 0.0, -1.0
            norm_x, norm_y = 1.0, 0.0
            base_x, base_y = -DSO, 0.0
            src_x_arr = base_x + p_arr * side_x
            src_y_arr = base_y + p_arr * side_y

        for iu in range(nu):

            u_det = u_arr[iu]

            # Ray x/y components independent of p and z
            rx = DSD * norm_x + u_det * side_x
            ry = DSD * norm_y + u_det * side_y

            for iv in range(nv):

                rz = v_arr[iv]

                rmag = np.sqrt(rx*rx + ry*ry + rz*rz)

                ray_x = rx / rmag
                ray_y = ry / rmag
                ray_z = rz / rmag

                abs_rx = abs(ray_x)
                abs_ry = abs(ray_y)
                abs_rz = abs(ray_z)

                inv_rx = 1.0 / ray_x if abs_rx > eps else 0.0
                inv_ry = 1.0 / ray_y if abs_ry > eps else 0.0
                inv_rz = 1.0 / ray_z if abs_rz > eps else 0.0

                for ip in range(ns_p):

                    src_x = src_x_arr[ip]
                    src_y = src_y_arr[ip]

                    # --- Bounding box intersection (fast form) ---
                    if abs_rx > eps:
                        tx1 = (bx_min - src_x) * inv_rx
                        tx2 = (bx_max - src_x) * inv_rx
                        tx_entry = min(tx1, tx2)
                        tx_exit  = max(tx1, tx2)
                    else:
                        tx_entry = -np.inf
                        tx_exit  =  np.inf

                    if abs_ry > eps:
                        ty1 = (by_min - src_y) * inv_ry
                        ty2 = (by_max - src_y) * inv_ry
                        ty_entry = min(ty1, ty2)
                        ty_exit  = max(ty1, ty2)
                    else:
                        ty_entry = -np.inf
                        ty_exit  =  np.inf

                    for izs in range(ns_z):

                        src_z = sz_arr[izs]

                        if abs_rz > eps:
                            tz1 = (bz_min - src_z) * inv_rz
                            tz2 = (bz_max - src_z) * inv_rz
                            tz_entry = min(tz1, tz2)
                            tz_exit  = max(tz1, tz2)
                        else:
                            tz_entry = -np.inf
                            tz_exit  =  np.inf

                        t_entry = max(tx_entry, ty_entry, tz_entry)
                        t_exit  = min(tx_exit, ty_exit, tz_exit)

                        if t_exit <= t_entry:
                            continue

                        # --- Entry voxel ---
                        ix = _calc_ir0(src_x, ray_x, t_entry, bx_min, bx_max, d_pix)
                        iy = _calc_ir0(src_y, ray_y, t_entry, by_min, by_max, d_pix)
                        iz = _calc_ir0(src_z, ray_z, t_entry, bz_min, bz_max, d_pix)

                        # --- Step init ---
                        ix_dir, tx_step, tx_next = _fp_step_init(
                            src_x, ix, ray_x, abs_rx, bx_min, d_pix)

                        iy_dir, ty_step, ty_next = _fp_step_init(
                            src_y, iy, ray_y, abs_ry, by_min, d_pix)

                        iz_dir, tz_step, tz_next = _fp_step_init(
                            src_z, iz, ray_z, abs_rz, bz_min, d_pix)

                        # --- Traverse ---
                        val = _aw_fp_traverse_3d(
                            img,
                            t_entry, t_exit,
                            tx_next, ty_next, tz_next,
                            tx_step, ty_step, tz_step,
                            ix, iy, iz,
                            ix_dir, iy_dir, iz_dir,
                            nx, ny, nz
                        )

                        sino[iside, ip, izs, iu, iv] = val * (DSD / rmag)

    return sino


@njit(fastmath=True, cache=True)
def aw_p_square4(img, sino, DSO, DSD, du, dv, d_pix):

    nx, ny, nz = img.shape
    ns, ns_p, ns_z, nu, nv = sino.shape

    # Image bounds
    bx_min, bx_max, _ = _img_bounds(nx, d_pix)
    by_min, by_max, _ = _img_bounds(ny, d_pix)
    bz_min, bz_max, _ = _img_bounds(nz, d_pix)

    # Detector coordinates
    u_arr = du * (np.arange(nu, dtype=np.float32) - (nu / 2 - 0.5))
    v_arr = dv * (np.arange(nv, dtype=np.float32) - (nv / 2 - 0.5))

    # Source sampling
    p_arr  = np.linspace(-DSO, DSO, ns_p).astype(np.float32)
    sz_arr = np.linspace(bz_min, bz_max, ns_z).astype(np.float32)

    for iside in range(ns):

        # --- Side geometry ---
        if iside == 0:
            side_x, side_y = 1.0, 0.0
            norm_x, norm_y = 0.0, 1.0
            base_x, base_y = 0.0, -DSO
        elif iside == 1:
            side_x, side_y = 0.0, 1.0
            norm_x, norm_y = -1.0, 0.0
            base_x, base_y = DSO, 0.0
        elif iside == 2:
            side_x, side_y = -1.0, 0.0
            norm_x, norm_y = 0.0, -1.0
            base_x, base_y = 0.0, DSO
        else:
            side_x, side_y = 0.0, -1.0
            norm_x, norm_y = 1.0, 0.0
            base_x, base_y = -DSO, 0.0

        # Precompute translated source arrays
        src_x_arr = base_x + p_arr * side_x
        src_y_arr = base_y + p_arr * side_y

        for iu in range(nu):

            u_det = u_arr[iu]

            # Ray components independent of source translation
            rx = DSD * norm_x + u_det * side_x
            ry = DSD * norm_y + u_det * side_y

            for iv in range(nv):

                rz = v_arr[iv]

                rmag = np.sqrt(rx*rx + ry*ry + rz*rz)

                ray_x = rx / rmag
                ray_y = ry / rmag
                ray_z = rz / rmag

                abs_rx = abs(ray_x)
                abs_ry = abs(ray_y)
                abs_rz = abs(ray_z)

                inv_rx = 1.0 / ray_x if abs_rx > eps else 0.0
                inv_ry = 1.0 / ray_y if abs_ry > eps else 0.0
                inv_rz = 1.0 / ray_z if abs_rz > eps else 0.0

                # -----------------------------
                # Compute base intersections
                # at reference source position
                # -----------------------------
                src_x_ref = base_x
                src_y_ref = base_y
                src_z_ref = 0.0

                if abs_rx > eps:
                    tx1 = (bx_min - src_x_ref) * inv_rx
                    tx2 = (bx_max - src_x_ref) * inv_rx
                    tx_entry_base = min(tx1, tx2)
                    tx_exit_base  = max(tx1, tx2)
                else:
                    tx_entry_base = -np.inf
                    tx_exit_base  =  np.inf

                if abs_ry > eps:
                    ty1 = (by_min - src_y_ref) * inv_ry
                    ty2 = (by_max - src_y_ref) * inv_ry
                    ty_entry_base = min(ty1, ty2)
                    ty_exit_base  = max(ty1, ty2)
                else:
                    ty_entry_base = -np.inf
                    ty_exit_base  =  np.inf

                if abs_rz > eps:
                    tz1 = (bz_min - src_z_ref) * inv_rz
                    tz2 = (bz_max - src_z_ref) * inv_rz
                    tz_entry_base = min(tz1, tz2)
                    tz_exit_base  = max(tz1, tz2)
                else:
                    tz_entry_base = -np.inf
                    tz_exit_base  =  np.inf

                # -----------------------------
                # Loop over translated sources
                # -----------------------------
                for ip in range(ns_p):

                    src_x = src_x_arr[ip]
                    src_y = src_y_arr[ip]

                    dx = src_x - src_x_ref
                    dy = src_y - src_y_ref

                    # Shift x and y intersections linearly
                    if abs_rx > eps:
                        tx_entry = tx_entry_base - dx * inv_rx
                        tx_exit  = tx_exit_base  - dx * inv_rx
                    else:
                        tx_entry = -np.inf
                        tx_exit  =  np.inf

                    if abs_ry > eps:
                        ty_entry = ty_entry_base - dy * inv_ry
                        ty_exit  = ty_exit_base  - dy * inv_ry
                    else:
                        ty_entry = -np.inf
                        ty_exit  =  np.inf

                    for izs in range(ns_z):

                        src_z = sz_arr[izs]
                        dz = src_z - src_z_ref

                        # Shift z intersections linearly
                        if abs_rz > eps:
                            tz_entry = tz_entry_base - dz * inv_rz
                            tz_exit  = tz_exit_base  - dz * inv_rz
                        else:
                            tz_entry = -np.inf
                            tz_exit  =  np.inf

                        t_entry = max(tx_entry, ty_entry, tz_entry)
                        t_exit  = min(tx_exit, ty_exit, tz_exit)

                        if t_exit <= t_entry:
                            continue

                        # Entry voxel
                        ix = _calc_ir0(src_x, ray_x, t_entry, bx_min, bx_max, d_pix)
                        iy = _calc_ir0(src_y, ray_y, t_entry, by_min, by_max, d_pix)
                        iz = _calc_ir0(src_z, ray_z, t_entry, bz_min, bz_max, d_pix)

                        # Step init
                        ix_dir, tx_step, tx_next = _fp_step_init(
                            src_x, ix, ray_x, abs_rx, bx_min, d_pix)

                        iy_dir, ty_step, ty_next = _fp_step_init(
                            src_y, iy, ray_y, abs_ry, by_min, d_pix)

                        iz_dir, tz_step, tz_next = _fp_step_init(
                            src_z, iz, ray_z, abs_rz, bz_min, d_pix)

                        # Traverse
                        val = _aw_fp_traverse_3d(
                            img,
                            t_entry, t_exit,
                            tx_next, ty_next, tz_next,
                            tx_step, ty_step, tz_step,
                            ix, iy, iz,
                            ix_dir, iy_dir, iz_dir,
                            nx, ny, nz
                        )

                        sino[iside, ip, izs, iu, iv] = val * (DSD / rmag)

    return sino

