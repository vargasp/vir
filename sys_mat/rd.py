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
    
    inv_dr = 1.0/dr
    
    
    if adr > 0.:
        tr_entry = (r_min - r0) * inv_dr
        tr_exit = (r_max - r0) * inv_dr
        return min(tr_entry, tr_exit), max(tr_entry, tr_exit)
    else:
        # Ray is (almost) vertical → no x-bound intersection
        return -np.inf, np.inf
                    

@njit(fastmath=True,inline='always',cache=True)
def _calc_ir0(ray_r_org,ray_r_hat,t_entry,img_bnd_r_min,nr,d_pix):

    r = ray_r_org + t_entry*ray_r_hat
    

    ir = int((r - img_bnd_r_min)/d_pix)
    if ir < 0:
        ir = 0
    elif ir >= nr:
        ir = nr - 1

    return ir


@njit(fastmath=True,inline='always',cache=True)
def _fp_step_init(r0, ir, dr, adr, r_min, d_pix):
    """
    Initialize Amanatides–Woo traversal parameters for ONE axis (x, y, or z).
    
    This function computes the quantities needed to traverse a regular
    Cartesian grid along a single axis using the Amanatides–Woo algorithm.
    
    The ray is assumed to be parameterized as:
        r(t) = r0 + t * d
    
    where:
        - r0 is the ray origin coordinate along this axis
        - dr is the ray direction component along this axis
        - t is the ray parameter (same t used for t_enter / t_exit)
    
    All returned values are expressed in terms of the ray parameter t
    (NOT voxel indices).
    
    Parameters
    ----------
    r0 : float
        Ray origin coordinate along this axis (img space).
        MUST be the same origin used to compute t_enter and t_exit.
    
    ir : int
        Integer voxel index along this axis at the ray entry point.
        Typically computed as:
            ir = floor((r_entry - r_min) / d_pix)
    
    dr : float
        Ray direction component along this axis.
    
    adr : float
        Absolute value of dr (i.e., abs(dr)).
        Passed in explicitly to avoid repeated abs() calls.
    
    r_min : float
        Img-coordinate location of the minimum grid boundary
        along this axis (voxel index 0).
    
    d_pix : float
        Voxel size along this axis.
    
    Returns
    -------
    idir : int
        Direction to step voxel indices along this axis:
            +1 if dr > 0 (increasing index),
            -1 if dr < 0 (decreasing index).
    
    tr_step : float
        Increment in ray parameter t between successive grid-plane
        crossings along this axis:
            tr_step = d_pix / |dr|.
    
    tr_next : float
        Ray parameter value at which the ray first crosses a grid plane
        along this axis.
        This is NOT a voxel index — it lives in ray-parameter space.
    """
    
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
    if adr > 0.0:
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
                    



@njit(fastmath=True, inline='always', cache=True)
def _fp_step_init_voxel(v0, iv, dv, adv):
    """
    Amanatides–Woo step initialization in voxel space.

    v(s) = v0 + s * dv
    Grid planes are at integer coordinates.
    """

    if dv > 0.0:
        idir = 1
        v_next_plane = iv + 1.0
    else:
        idir = -1
        v_next_plane = iv * 1.0

    if dv != 0.0:
        t_step = 1.0 / adv
        t_next = (v_next_plane - v0) / dv
    else:
        t_step = np.inf
        t_next = np.inf

    return idir, t_step, t_next


@njit(fastmath=True, inline='always', cache=True)
def _aw_fp_traverse_3d_voxel(img,
                             t_entry, t_exit,
                             tx_next, ty_next, tz_next,
                             tx_step, ty_step, tz_step,
                             ix, iy, iz,
                             ix_dir, iy_dir, iz_dir,
                             nx, ny, nz,
                             ray_length):

    t = t_entry
    acc = np.float32(0.0)

    while True:

        if t >= t_exit:
            break

        if ix < 0 or iy < 0 or iz < 0 or \
           ix >= nx or iy >= ny or iz >= nz:
            break

        # Find next crossing
        t_next = tx_next
        if ty_next < t_next:
            t_next = ty_next
        if tz_next < t_next:
            t_next = tz_next

        if t_next > t_exit:
            t_next = t_exit

        dt = t_next - t

        if dt > 0.0:
            acc += img[ix, iy, iz] * dt * ray_length

        t = t_next

        if t >= t_exit:
            break

        # Robust multi-axis tie handling
        if t_next == tx_next:
            ix += ix_dir
            tx_next += tx_step

        if t_next == ty_next:
            iy += iy_dir
            ty_next += ty_step

        if t_next == tz_next:
            iz += iz_dir
            tz_next += tz_step

    return acc



@njit(fastmath=True,inline='always',cache=True)
def _aw_fp_traverse_2d(iu, img,t_entry,t_exit,tx_next,ty_next,tx_step,ty_step,
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

    #print(type(t),type(t_exit))
    while t < t_exit:
        #print(ix,iy,t,t_exit)
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
        print(ix,iy,iz,t,t_exit)

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


@njit(fastmath=True, cache=True)
def _joseph_fp_3d_voxel(img,
                        t_entry, t_exit, t_step,
                        v0x, v0y, v0z,
                        dvx, dvy, dvz,
                        ray_length):

    nx, ny, nz = img.shape

    acc = np.float32(0.0)
    t = t_entry

    while t <= t_exit:

        # Voxel-space position
        x = v0x + t * dvx
        y = v0y + t * dvy
        z = v0z + t * dvz

        ix = int(x)
        iy = int(y)
        iz = int(z)

        if 0 <= ix < nx-1 and \
           0 <= iy < ny-1 and \
           0 <= iz < nz-1:

            dx = x - ix
            dy = y - iy
            dz = z - iz

            # Trilinear interpolation
            c000 = img[ix  , iy  , iz  ]
            c100 = img[ix+1, iy  , iz  ]
            c010 = img[ix  , iy+1, iz  ]
            c110 = img[ix+1, iy+1, iz  ]
            c001 = img[ix  , iy  , iz+1]
            c101 = img[ix+1, iy  , iz+1]
            c011 = img[ix  , iy+1, iz+1]
            c111 = img[ix+1, iy+1, iz+1]

            c00 = c000 + dx * (c100 - c000)
            c10 = c010 + dx * (c110 - c010)
            c01 = c001 + dx * (c101 - c001)
            c11 = c011 + dx * (c111 - c011)

            c0 = c00 + dy * (c10 - c00)
            c1 = c01 + dy * (c11 - c01)

            val = c0 + dz * (c1 - c0)

            acc += val * t_step * ray_length

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
        ix = int(fx)
        iy = int(fy)
        iz = int(fz)

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


@njit(fastmath=True,cache=True)
def aw_fp_par_2d(img, ang_arr, nu, du=1.0, su=0.0, d_pix=1.0,joseph=False):
    """
    2D parallel-beam forward projection using Amanatides–Woo ray traversal.

    Computes the line integrals of a 2D image over a set of parallel-beam
    rays defined by projection angles and detector positions.

    Geometry:
        - Image is defined on a Cartesian grid of size (nX, nY)
        - Pixel size is dPix
        - Coordinate system is centered at (0, 0)
        - For angle θ:
            ray direction = (cos θ, sin θ)
            detector offset = det
            ray origin = (-sin θ * det, cos θ * det)

    The projection is computed as:
        sino[ia, idt] = ∫ img(x, y) ds
    where the integral is approximated by summing voxel values weighted
    by exact intersection lengths.

    Parameters
    ----------
    img : ndarray, shape (nX, nY)
        Input image (voxel values).
    ang_arr : ndarray, shape (nang,)
        Projection angles in radians.
    nu : int
        Number of detector bins.
    du : float, optional
        Detector spacing (default: 1.0).
    d_pix : float, optional
        Pixel size (default: 1.0).

    Returns
    -------
    sino : ndarray, shape (nAng, n_dets)
        Sinogram (line integrals).
    """

    # number of pixels in x and y
    nx, ny = img.shape         

    # Output sinogram: (angle, detector)
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    # Define image bounds in world coordinates
    img_bnd_x_min, img_bnd_x_max, x0 = _img_bounds(nx,d_pix)
    img_bnd_y_min, img_bnd_y_max, y0 = _img_bounds(ny,d_pix)
    step=.5          


    # Detector bin centers (detector coordinate u)
    u_arr = du*(np.arange(nu) - nu/2 + 0.5 + su)

    # Precompute ray direction for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    # Main loops: angles → detectors → voxel traversal
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):
        
        #Ray directions in unit vector components
        # Ray direction (parallel abd orthoganal to detector)
        det_u_orn = (-sin_ang, cos_ang)

        ray_x_hat = cos_ang
        ray_y_hat = sin_ang 
        
        # Absolute values are used for step size calculations
        rx_abs = abs(ray_x_hat)
        ry_abs = abs(ray_y_hat)

        for iu, u in enumerate(u_arr):
            # Each ray is parameterized as:
            # r(t) = (rx_u, ry_u) + t * (rx, ry)
            
            # Ray origination at the detector
            ray_x_org = u*det_u_orn[0]
            ray_y_org = u*det_u_orn[1]

            # Intersect ray with image bounding box
            tx_min, tx_max = _intersect_bounding(ray_x_org, ray_x_hat, rx_abs, img_bnd_x_min, img_bnd_x_max)
            ty_min, ty_max = _intersect_bounding(ray_y_org, ray_y_hat, ry_abs, img_bnd_y_min, img_bnd_y_max)

            # Combined entry/exit interval
            t_entry = max(tx_min, ty_min)
            t_exit = min(tx_max,  ty_max)

            print(type(t_entry), type(t_exit))

            if t_exit <= t_entry:
                # Ray does not intersect the image
                continue

            if joseph:
                sino[ia, iu] = _joseph_fp_2d(img,t_entry,t_exit,step,
                                             ray_x_org,ray_y_org,
                                             ray_x_hat,ray_y_hat,x0,y0,d_pix)

            else:
                

                ix_entry = _calc_ir0(ray_x_org,ray_x_hat,t_entry,img_bnd_x_min,nx,d_pix)
                iy_entry = _calc_ir0(ray_y_org,ray_y_hat,t_entry,img_bnd_y_min,ny,d_pix)

                print(ix_entry,iy_entry)

                # Amanatides–Woo stepping initialization
                ix_dir, tx_step, tx_next = _fp_step_init(ray_x_org,ix_entry,ray_x_hat, rx_abs, img_bnd_x_min, d_pix)
                iy_dir, ty_step, ty_next = _fp_step_init(ray_y_org,iy_entry,ray_y_hat, ry_abs, img_bnd_y_min, d_pix)
                
                
                # Traverse the grid voxel-by-voxel
                sino[ia, iu] = _aw_fp_traverse_2d(iu, img,t_entry,t_exit,
                                     tx_next, ty_next, tx_step, ty_step,
                                     ix_entry,iy_entry,ix_dir,iy_dir,nx,ny,)

    #Returns the sino
    return sino

                    
@njit(fastmath=True,cache=True)
def aw_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=1.0, su=0.0, d_pix=1.0,joseph=False):
    """
    2D flat-panel fan-beam forward projection using Amanatides–Woo traversal.

    Parameters
    ----------
    img : ndarray (nX, nY)
        Input image.
    Angs : ndarray (nAng,)
        Projection angles in radians.
    n_dets : int
        Number of detector elements.
    DSO : float
        Distance from source to isocenter.
    DSD : float
        Distance from source to detector.
    du : float
        Detector spacing.
    d_pix : float
        Pixel size.

    Returns
    -------
    sino : ndarray (nAng, nDets)
        Fan-beam sinogram.
    """

    # number of pixels in x and y
    nx, ny = img.shape

    # Output sinogram: (angle, detector)
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    # Define image bounds in world coordinates
    # Grid: centered at zero, units in physical space
    #Joseph parameters - first pixel
    img_bnd_x_min, img_bnd_x_max, x0 = _img_bounds(nx,d_pix)
    img_bnd_y_min, img_bnd_y_max, y0 = _img_bounds(ny,d_pix)

    #Joseph parameters
    step=.5

    # Detector bin centers
    u_arr = du*(np.arange(nu) - nu/2 + 0.5 + su)

    # Precompute ray direction for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    # Main loops: angles → detectors → voxel traversal
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):

        # Ray origination at the source
        ray_x_org = DSO * cos_ang
        ray_y_org = DSO * sin_ang

        # Detector reference point
        det_x_org = -(DSD - DSO)*cos_ang
        det_y_org = -(DSD - DSO)*sin_ang

        # Ray direction (parallel abd orthoganal to detector)
        det_u_orn = (-sin_ang, cos_ang)

        for iu, u in enumerate(u_arr):
            # Each ray is parameterized as:
            # r(t) = (ray_o_x_pos, ray_o_y_pos) + t * (rx, ry)

            # Detector point
            det_x = det_x_org + u*det_u_orn[0]
            det_y = det_y_org + u*det_u_orn[1]

            # Ray from source to detector
            ray_x_vec = det_x - ray_x_org
            ray_y_vec = det_y - ray_y_org

            #Ray directions in unit vector components
            ray_mag = np.sqrt(ray_x_vec**2 + ray_y_vec**2)
            ray_x_hat = ray_x_vec/ray_mag
            ray_y_hat = ray_y_vec/ray_mag

            # Absolute values are used for step size calculations
            rx_abs = abs(ray_x_hat)
            ry_abs = abs(ray_y_hat)

            # Intersection with image bounding box
            t_x_min, t_x_max = _intersect_bounding(ray_x_org, ray_x_hat, rx_abs, img_bnd_x_min, img_bnd_x_max)
            t_y_min, t_y_max = _intersect_bounding(ray_y_org, ray_y_hat, ry_abs, img_bnd_y_min, img_bnd_y_max)

            t_entry = max(t_x_min, t_y_min)
            t_exit = min(t_x_max, t_y_max)

            if t_exit <= t_entry:
                continue

            if joseph:    
                sino[ia, iu] = _joseph_fp_2d(img,t_entry,t_exit,step,
                                             ray_x_org,ray_y_org,
                                             ray_x_hat,ray_y_hat,x0,y0,d_pix)
            else:
       
                ix_entry = _calc_ir0(ray_x_org,ray_x_hat,t_entry,img_bnd_x_min,nx,d_pix)
                iy_entry = _calc_ir0(ray_y_org,ray_y_hat,t_entry,img_bnd_y_min,ny,d_pix)
    
                # Amanatides–Woo stepping initialization
                ix_dir, tx_step, tx_next = _fp_step_init(ray_x_org, ix_entry, ray_x_hat, rx_abs, img_bnd_x_min, d_pix)
                iy_dir, ty_step, ty_next = _fp_step_init(ray_y_org, iy_entry, ray_y_hat, ry_abs, img_bnd_y_min, d_pix)
                
                # Grid traversal
                # Store final line integral
                sino[ia, iu] = _aw_fp_traverse_2d(iu,img,t_entry,t_exit,
                                     tx_next,ty_next,tx_step,ty_step,
                                     ix_entry,iy_entry,ix_dir,iy_dir,nx,ny)
                
    return sino




@njit(fastmath=True,cache=True)
def aw_bp_par_2d(sino, ang_arr, img_shape, du=1.0, su=0.0, d_pix=1.0, joseph=False):
    """
    2D parallel-beam ray-driven back-projection.
    Parameters identical to forward AW-projection, but swaps image and sinogram roles.
    """
    nx, ny = img_shape
    na, nu = sino.shape
    img = np.zeros((nx, ny), dtype=np.float32)

    # Define image bounds in world coordinates
    img_bnd_x_min, img_bnd_x_max, x0 = _img_bounds(nx,d_pix)
    img_bnd_y_min, img_bnd_y_max, y0 = _img_bounds(ny,d_pix)
    step=.1

    u_arr = du * (np.arange(nu) - nu/2 + 0.5 + su)
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
                
                
                _joseph_bp_2d(img, d_pix, sino[ia,iu], cos_ang, sin_ang, ray_x_org, ray_y_org,
                           x0, y0, t_entry, t_exit, step)
            else:
                
                #ix_entry = _calc_ir0(ray_x_org,ray_x_hat,t_entry,img_bnd_x_min,img_bnd_x_max,d_pix)
                #iy_entry = _calc_ir0(ray_y_org,ray_y_hat,t_entry,img_bnd_y_min,img_bnd_y_max,d_pix)
                
                ix_entry = _calc_ir0(ray_x_org,ray_x_hat,t_entry,img_bnd_x_min,nx,d_pix)
                iy_entry = _calc_ir0(ray_y_org,ray_y_hat,t_entry,img_bnd_y_min,ny,d_pix)

                
                
                ix_dir, tx_step, tx_next = _fp_step_init(ray_x_org, ix_entry, ray_x_hat, rx_abs, img_bnd_x_min, d_pix)
                iy_dir, ty_step, ty_next = _fp_step_init(ray_y_org, iy_entry, ray_y_hat, ry_abs, img_bnd_y_min, d_pix)
                _aw_bp_traverse_2d(img, sino[ia, iu],t_entry,t_exit,
                            tx_next,ty_next,tx_step,ty_step, 
                            ix_entry,iy_entry,ix_dir,iy_dir,nx,ny)
                
                
    return img / na / d_pix / d_pix * du


@njit(fastmath=True,cache=True)
def aw_bp_fan_2d(sino, ang_arr, img_shape, DSO, DSD, du=1.0, su=0.0, d_pix=1.0, joseph=False):
    """
    2D fan-beam ray-driven back-projection with flat panel geometry.
    """
    nx, ny = img_shape
    na, nu = sino.shape
    img = np.zeros((nx, ny), dtype=np.float32)

    img_bnd_x_min, img_bnd_x_max, x0 = _img_bounds(nx,d_pix)
    img_bnd_y_min, img_bnd_y_max, y0 = _img_bounds(ny,d_pix)
    
    step = .5

    u_arr = du*(np.arange(nu) - nu/2 + 0.5 + su)
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    for ia, (cos_ang, sin_ang) in enumerate(zip(cos_ang_arr, sin_ang_arr)):
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
            
            rx_abs = abs(ray_x_hat)
            ry_abs = abs(ray_y_hat)

            tx_entry, tx_exit = _intersect_bounding(ray_x_org, ray_x_hat, rx_abs, img_bnd_x_min, img_bnd_x_max)
            ty_entry, ty_exit = _intersect_bounding(ray_y_org, ray_y_hat, ry_abs, img_bnd_y_min, img_bnd_y_max)

            t_entry = max(tx_entry, ty_entry)
            t_exit  = min(tx_exit, ty_exit)

            if t_exit <= t_entry:
                continue

            if joseph:
                _joseph_bp_2d(img, d_pix, sino[ia,iu], ray_x_hat, ray_y_hat, ray_x_org, ray_y_org,
                           x0, y0, t_entry, t_exit, step)
            else:
                    
                ix_entry = _calc_ir0(ray_x_org,ray_x_hat,t_entry,img_bnd_x_min,nx,d_pix)
                iy_entry = _calc_ir0(ray_y_org,ray_y_hat,t_entry,img_bnd_y_min,ny,d_pix)

    
                # Amanatides–Woo stepping initialization
                ix_dir, tx_step, tx_next = _fp_step_init(ray_x_org, ix_entry, ray_x_hat, rx_abs, img_bnd_x_min, d_pix)
                iy_dir, ty_step, ty_next = _fp_step_init(ray_y_org, iy_entry, ray_y_hat, ry_abs, img_bnd_y_min, d_pix)
  
                _aw_bp_traverse_2d(img, sino[ia, iu],t_entry,t_exit,
                                   tx_next,ty_next,tx_step,ty_step,
                                   ix_entry,iy_entry,ix_dir,iy_dir,nx,ny)

    return img / na / d_pix / d_pix * du



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
def aw_p_cone_3dO(img,sino,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,joseph,bp):

    nx, ny, nz = img.shape
    na, nu, nv = sino.shape
    
    inv_d_pix = 1.0 / d_pix

    
    img_bnd_x_min, img_bnd_x_max, x0 = _img_bounds(nx,d_pix)
    img_bnd_y_min, img_bnd_y_max, y0 = _img_bounds(ny,d_pix)
    img_bnd_z_min, img_bnd_z_max, z0 = _img_bounds(nz,d_pix)
    step = np.float32(0.5)

    u_arr = du*(np.arange(nu,dtype=np.float32) - np.float32(nu/2 - 0.5))
    v_arr = dv*(np.arange(nv,dtype=np.float32) - np.float32(nu/2 - 0.5))

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

                ray_length = np.sqrt(ray_x_vec**2 +
                                     ray_y_vec**2 +
                                     ray_z_vec**2)

                ray_x_org_v = (ray_x_org - img_bnd_x_min) * inv_d_pix
                ray_y_org_v = (ray_y_org - img_bnd_y_min) * inv_d_pix
                ray_z_org_v = (ray_z_org - img_bnd_z_min) * inv_d_pix
                
                ray_x_vec_v = ray_x_vec * inv_d_pix
                ray_y_vec_v = ray_y_vec * inv_d_pix
                ray_z_vec_v = ray_z_vec * inv_d_pix

                tx_entry, tx_exit = _intersect_bounding(
                    ray_x_org_v, ray_x_vec_v, abs(ray_x_vec_v),
                    0.0, nx)
                
                ty_entry, ty_exit = _intersect_bounding(
                    ray_y_org_v, ray_y_vec_v, abs(ray_y_vec_v),
                    0.0, ny)
                
                tz_entry, tz_exit = _intersect_bounding(
                    ray_z_org_v, ray_z_vec_v, abs(ray_z_vec_v),
                    0.0, nz)

                print(tz_entry,tz_exit)

                """
                ray_mag = np.sqrt(ray_x_vec**2 + ray_y_vec**2 + ray_z_vec**2)
                ray_x_hat = ray_x_vec/ray_mag
                ray_y_hat = ray_y_vec/ray_mag
                ray_z_hat = ray_z_vec/ray_mag

                tx_entry,tx_exit = _intersect_bounding(ray_x_org, ray_x_hat, abs(ray_x_hat), img_bnd_x_min, img_bnd_x_max)
                ty_entry,ty_exit = _intersect_bounding(ray_y_org, ray_y_hat, abs(ray_y_hat), img_bnd_y_min, img_bnd_y_max)
                tz_entry,tz_exit = _intersect_bounding(ray_z_org, ray_z_hat, abs(ray_z_hat), img_bnd_z_min, img_bnd_z_max)
                """
                
                t_entry = max(tx_entry, ty_entry, tz_entry)
                t_exit  = min(tx_exit, ty_exit, tz_exit)


                if t_exit <= t_entry:
                    continue

                if joseph:

                    if bp:
                        
                        """
                        _joseph_bp_3d(img,sino[ia,iu,iv],t_entry,t_exit,step,
                                                       ray_x_org,ray_y_org,ray_z_org,
                                                       ray_x_hat,ray_y_hat,ray_z_hat,
                                                       x0,y0,z0,d_pix)
                        
                        """
                        print("not implmented")

                        
                        
                    else:
                        """
                        sino[ia,iu,iv] = _joseph_fp_3d(img,t_entry,t_exit,step,
                                                   ray_x_org,ray_y_org,ray_z_org,
                                                   ray_x_hat,ray_y_hat,ray_z_hat,
                                                   x0,y0,z0,d_pix)
                        
                        """
                        
                        sino[ia, iu, iv] = _joseph_fp_3d_voxel(
    img,
    t_entry, t_exit,
    step,  # same sampling step
    ray_x_org_v,
    ray_y_org_v,
    ray_z_org_v,
    ray_x_vec_v,
    ray_y_vec_v,
    ray_z_vec_v,
    ray_length)
                else:

                    #ix_entry = _calc_ir0(ray_x_org,ray_x_hat,t_entry,img_bnd_x_min,nx,d_pix)
                    #iy_entry = _calc_ir0(ray_y_org,ray_y_hat,t_entry,img_bnd_y_min,ny,d_pix)
                    #iz_entry = _calc_ir0(ray_z_org,ray_z_hat,t_entry,img_bnd_z_min,nz,d_pix)
                    #ix_dir,tx_step,tx_next = _fp_step_init(ray_x_org,ix_entry, ray_x_hat, abs(ray_x_hat), img_bnd_x_min, d_pix)
                    #iy_dir,ty_step,ty_next = _fp_step_init(ray_y_org,iy_entry, ray_y_hat, abs(ray_y_hat), img_bnd_y_min, d_pix)
                    #iz_dir,tz_step,tz_next = _fp_step_init(ray_z_org,iz_entry, ray_z_hat, abs(ray_z_hat), img_bnd_z_min, d_pix)


                    ix_entry = int(ray_x_org_v + t_entry * ray_x_vec_v)
                    iy_entry = int(ray_y_org_v + t_entry * ray_y_vec_v)
                    iz_entry = int(ray_z_org_v + t_entry * ray_z_vec_v)

                    
                    ix_dir, tx_step, tx_next = _fp_step_init_voxel(
                        ray_x_org_v + t_entry * ray_x_vec_v,
                        ix_entry,
                        ray_x_vec_v,
                        abs(ray_x_vec_v))
                    
                    iy_dir, ty_step, ty_next = _fp_step_init_voxel(
                        ray_y_org_v + t_entry * ray_y_vec_v,
                        iy_entry,
                        ray_y_vec_v,
                        abs(ray_y_vec_v))
                    
                    iz_dir, tz_step, tz_next = _fp_step_init_voxel(
                        ray_z_org_v + t_entry * ray_z_vec_v,
                        iz_entry,
                        ray_z_vec_v,
                        abs(ray_z_vec_v))


                    if bp:
                        _aw_bp_traverse_3d(img,sino[ia,iu,iv],t_entry,t_exit,
                            tx_next,ty_next,tz_next,tx_step, ty_step, tz_step,
                            ix_entry, iy_entry, iz_entry,
                            ix_dir,iy_dir,iz_dir,nx,ny,nz)

                    else:
                        """
                        sino[ia,iu,iv] = _aw_fp_traverse_3d(img,t_entry, t_exit,
                            tx_next,ty_next,tz_next,tx_step, ty_step, tz_step,
                            ix_entry, iy_entry, iz_entry,
                            ix_dir,iy_dir,iz_dir,nx,ny,nz)
                                        
                        ray_scale = DSD/np.sqrt(DSD**2 + u**2 + v**2)
                        sino[ia, iu, iv]= sino[ia, iu, iv]*ray_scale
                        """
                        sino[ia, iu, iv] = _aw_fp_traverse_3d_voxel(
                                            img,
                                            t_entry, t_exit,
                                            tx_next, ty_next, tz_next,
                                            tx_step, ty_step, tz_step,
                                            ix_entry, iy_entry, iz_entry,
                                            ix_dir, iy_dir, iz_dir,
                                            nx, ny, nz,
                                            ray_length)

                    
    if bp:
        return img/ang_arr.size*du*dv
    else:
        return sino
    
    
    
@njit(fastmath=True, cache=True)
def aw_p_cone_3d(img, sino, ang_arr, DSO, DSD, du, dv, su, sv, d_pix,
                 joseph=False, bp=False):

    nx, ny, nz = img.shape
    na, nu, nv = sino.shape

    # Voxel-space bounds
    inv_d_pix = 1.0 / d_pix

    # Image bounds in world coordinates
    img_bnd_x_min, img_bnd_x_max, _ = -d_pix*nx/2, d_pix*nx/2, 0.0
    img_bnd_y_min, img_bnd_y_max, _ = -d_pix*ny/2, d_pix*ny/2, 0.0
    img_bnd_z_min, img_bnd_z_max, _ = -d_pix*nz/2, d_pix*nz/2, 0.0

    # Detector positions
    u_arr = du*(np.arange(nu, dtype=np.float32) - np.float32(nu/2 - 0.5) + su)
    v_arr = dv*(np.arange(nv, dtype=np.float32) - np.float32(nv/2 - 0.5) + sv)

    cos_arr = np.cos(ang_arr)
    sin_arr = np.sin(ang_arr)

    step = np.float32(0.5)  # Joseph step, later can adapt to voxel step

    for ia in range(na):
        cos_ang = cos_arr[ia]
        sin_ang = sin_arr[ia]

        # Source position
        ray_x_org = DSO * cos_ang
        ray_y_org = DSO * sin_ang
        ray_z_org = 0.0

        # Detector origin (center)
        det_x_org = -(DSD - DSO) * cos_ang
        det_y_org = -(DSD - DSO) * sin_ang
        det_z_org = 0.0

        # Detector u,v positions
        det_x_arr = det_x_org + u_arr * -sin_ang
        det_y_arr = det_y_org + u_arr * cos_ang

        for iv in range(nv):
            v = v_arr[iv]
            det_z = det_z_org + v

            for iu in range(nu):
                det_x = det_x_arr[iu]
                det_y = det_y_arr[iu]

                # Ray vector in world space
                ray_vec_x = det_x - ray_x_org
                ray_vec_y = det_y - ray_y_org
                ray_vec_z = det_z - ray_z_org

                ray_length = np.sqrt(ray_vec_x**2 + ray_vec_y**2 + ray_vec_z**2)

                # Transform to voxel space
                v0x = (ray_x_org - img_bnd_x_min) * inv_d_pix
                v0y = (ray_y_org - img_bnd_y_min) * inv_d_pix
                v0z = (ray_z_org - img_bnd_z_min) * inv_d_pix

                dvx = ray_vec_x * inv_d_pix
                dvy = ray_vec_y * inv_d_pix
                dvz = ray_vec_z * inv_d_pix

                # Compute voxel-space slab intersections
                tmin_x = (0.0 - v0x)/dvx if dvx != 0 else -np.inf
                tmax_x = (nx - v0x)/dvx if dvx != 0 else np.inf
                if tmin_x > tmax_x:
                    tmin_x, tmax_x = tmax_x, tmin_x

                tmin_y = (0.0 - v0y)/dvy if dvy != 0 else -np.inf
                tmax_y = (ny - v0y)/dvy if dvy != 0 else np.inf
                if tmin_y > tmax_y:
                    tmin_y, tmax_y = tmax_y, tmin_y

                tmin_z = (0.0 - v0z)/dvz if dvz != 0 else -np.inf
                tmax_z = (nz - v0z)/dvz if dvz != 0 else np.inf
                if tmin_z > tmax_z:
                    tmin_z, tmax_z = tmax_z, tmin_z

                t_entry = max(tmin_x, tmin_y, tmin_z)
                t_exit  = min(tmax_x, tmax_y, tmax_z)

                if t_exit <= t_entry:
                    continue  # Ray misses volume

                # Initial voxel indices
                ix = int(v0x + t_entry * dvx)
                iy = int(v0y + t_entry * dvy)
                iz = int(v0z + t_entry * dvz)

                # Clamp indices
                ix = max(0, min(nx-1, ix))
                iy = max(0, min(ny-1, iy))
                iz = max(0, min(nz-1, iz))

                # Step initialization
                ix_dir, tx_step, tx_next = _fp_step_init_voxel(v0x + t_entry*dvx, ix, dvx, abs(dvx))
                iy_dir, ty_step, ty_next = _fp_step_init_voxel(v0y + t_entry*dvy, iy, dvy, abs(dvy))
                iz_dir, tz_step, tz_next = _fp_step_init_voxel(v0z + t_entry*dvz, iz, dvz, abs(dvz))

                if joseph:
                    sino[ia, iu, iv] = _joseph_fp_3d_voxel(
                        img, t_entry, t_exit, step,
                        v0x, v0y, v0z, dvx, dvy, dvz, ray_length)
                else:
                    sino[ia, iu, iv] = _aw_fp_traverse_3d_voxel(
                        img, t_entry, t_exit,
                        tx_next, ty_next, tz_next,
                        tx_step, ty_step, tz_step,
                        ix, iy, iz,
                        ix_dir, iy_dir, iz_dir,
                        nx, ny, nz,
                        ray_length)

    if bp:
        return img / na * du * dv
    else:
        return sino

