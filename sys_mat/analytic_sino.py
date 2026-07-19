# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:03:35 2026

@author: varga
"""

import numpy as np


from itertools import combinations
from scipy.spatial import ConvexHull,QhullError
from scipy.integrate import lebedev_rule


EPS = 1e-9


#########################################
#Numercial Phantoms - Discretized Spheres
#########################################

def _discretized_sphere_exact_voxel(x0,x1,y0,y1,z0,z1,r,ns=8):
    """
    Compute the volume of intersection between a sphere and an axis-aligned cube
    (voxel) using Gauss-Legendre quadrature.

    The sphere is assumed to be centered at the origin:
        x^2 + y^2 + z^2 <= r^2

    and the cube is defined by the Cartesian bounds:
        x0 <= x <= x1
        y0 <= y <= y1
        z0 <= z <= z1

    The method performs numerical integration over the x-y plane and computes
    the exact z-overlap analytically for each quadrature sample point.

    Parameters
    ----------
    x0, x1 : float
        Lower and upper x-bounds of the cube.

    y0, y1 : float
        Lower and upper y-bounds of the cube.

    z0, z1 : float
        Lower and upper z-bounds of the cube.

    r : float
        Sphere radius.

    ns : int, optional
        Order of Gauss-Legendre quadrature.
        Typical values:
            4  -> moderate accuracy
            8  -> very accurate
            12 -> near machine precision

    Returns
    -------
    vol : float
        Volume of the sphere contained inside the cube.

    Notes
    -----
    The sphere equation is:
        x^2 + y^2 + z^2 = r^2

    For fixed (x,y), the sphere spans:
        -sqrt(r^2 - x^2 - y^2) <= z <= sqrt(r^2 - x^2 - y^2)

    The algorithm integrates this z-overlap over the cube cross-section.
    """

    # Obtain Gauss-Legendre quadrature nodes and weights on [-1,1]
    xs, ws = np.polynomial.legendre.leggauss(ns)

    # Map quadrature nodes from [-1,1] -> [x0,x1]
    # Standard affine transform:
    #   x_mapped = 0.5*(x+1)*(b-a) + a
    # Weights scale by interval length / 2.
    xs = 0.5 * (xs + 1) * (x1 - x0) + x0
    wx = ws * (x1 - x0) / 2

    # Generate quadrature nodes and weights for y integration
    ys = 0.5 * (
        xs * 0 + np.polynomial.legendre.leggauss(ns)[0] + 1
    ) * (y1 - y0) + y0

    wy = np.polynomial.legendre.leggauss(ns)[1] * (y1 - y0) / 2

    # Accumulator for total intersection volume
    vol = 0.0

    # Perform 2D quadrature over x-y plane
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):

            # Squared radial distance from sphere center
            d2 = x * x + y * y

            # If outside sphere cross-section, skip
            if d2 >= r * r:
                continue

            # Sphere z extent at this (x,y):
            #     z = ±sqrt(r^2 - x^2 - y^2)
            z = np.sqrt(r * r - d2)

            # Compute overlap between:
            # sphere interval : [-z, +z]
            # voxel interval  : [z0, z1]
            # using interval clipping.
            zlo = max(z0, -z)
            zhi = min(z1,  z)

            # Positive overlap contributes volume
            if zhi > zlo:

                # Quadrature contribution:
                #   weight_x * weight_y * overlap_height
                vol += wx[i] * wy[j] * (zhi - zlo)

    return vol


def discretized_sphere_exact(s, nx, ny, nz):
    """
    Generate a voxelized sphere phantom using exact sphere-voxel intersection
    volumes.

    Each voxel value equals the fraction of voxel volume occupied by the sphere.
    This produces an anti-aliased sphere phantom with highly accurate boundary
    representation.

    Parameters
    ----------
    s : array-like
        Sphere parameters:

            [x0, y0, z0, r]

        where:
            x0, y0, z0 : float
                Sphere center in voxel coordinates.

            r : float
                Sphere radius in voxel units.

    nx, ny, nz : int
        Number of voxels along x, y, and z dimensions.

    Returns
    -------
    out : ndarray
        3D array of shape (nx, ny, nz) containing voxel occupancy values.

    Notes
    -----
    The volume grid is centered at:

        (nx/2, ny/2, nz/2)

    so voxel coordinates are shifted relative to the sphere center before
    computing intersection volumes.
    """

    # Sphere center and radius
    x0, y0, z0, r = s

    # Output phantom volume
    out = np.zeros((nx, ny, nz), dtype=float)

    # Iterate over every voxel in the 3D grid
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Compute voxel bounds in sphere-centered coordinates
                # Each voxel spans one unit:
                #   [i, i+1]
                #
                # shifted so that the image center corresponds to zero.
                xlo = i - nx / 2 - x0
                xhi = xlo + 1

                ylo = j - ny / 2 - y0
                yhi = ylo + 1

                zlo = k - nz / 2 - z0
                zhi = zlo + 1

                # Compute exact sphere-volume overlap for this voxel
                out[i, j, k] = _discretized_sphere_exact_voxel(
                    xlo, xhi,
                    ylo, yhi,
                    zlo, zhi,
                    r
                )

    return out


def discretized_sphere_approx(s, nx, ny, nz):

    """
    Generate an approximate voxelized sphere using a smooth occupancy model.

    Unlike an exact sphere voxelization based on sphere-voxel intersection
    volumes, this method estimates voxel occupancy using the signed distance
    from voxel centers to the sphere surface.

    The result is a smooth anti-aliased sphere representation where:
        occ = 1   -> voxel fully inside sphere
        occ = 0   -> voxel fully outside sphere
        0 < occ < 1 -> partial occupancy near boundary

    Parameters
    ----------
    s : array-like
        Sphere parameters:
            [x0, y0, z0, r]

        where:
            x0, y0, z0 : float
                Sphere center in voxel coordinates.

            r : float
                Sphere radius in voxel units.

    nx, ny, nz : int
        Number of voxels along x, y, and z dimensions.

    Returns
    -------
    occ : ndarray
        Array of shape (nx, ny, nz) containing approximate occupancy
        values in the range [0,1].

    Notes
    -----
    The approximation is based on a linear ramp around the sphere boundary
    using the signed distance field:

        d = ||x|| - r

    where:
        d < 0 : inside sphere
        d > 0 : outside sphere

    The transition width is chosen as half the voxel diagonal:
        sqrt(3)/2

    which produces a smooth subvoxel boundary approximation.
    """

    # Sphere center and radius
    x0, y0, z0, r = s

    # Generate voxel-center coordinate grids.
    # The +0.5 places coordinates at voxel centers.
    # Coordinates are shifted so that:
    #   image center -> (0,0,0)
    #
    # and then translated relative to the sphere center.
    x = np.arange(nx)[:, None, None] + 0.5 - nx / 2 - x0
    y = np.arange(ny)[None, :, None] + 0.5 - ny / 2 - y0
    z = np.arange(nz)[None, None, :] + 0.5 - nz / 2 - z0

    # Signed distance from voxel centers to sphere surface.
    #   d < 0 : inside sphere
    #   d = 0 : on surface
    #   d > 0 : outside sphere
    d = np.sqrt(x * x + y * y + z * z) - r

    # Transition width used for soft occupancy interpolation.
    # Half the voxel diagonal:
    #   sqrt(1^2 + 1^2 + 1^2) / 2 = sqrt(3)/2
    #
    # This approximates subvoxel partial-volume effects.
    w = np.sqrt(3) / 2

    # Convert signed distance into occupancy fraction.
    # Interior voxels approach 1.
    # Exterior voxels approach 0.
    #
    # Boundary voxels transition linearly across width w.
    occ = np.clip(0.5 - d / w, 0, 1)

    # Store as float32 to reduce memory usage
    return occ.astype(np.float32)


##########################################################
#Analytic Sinograms - Line Integration - Circles / Spheres
##########################################################

def analytic_sino_circle_par(s,ang,u):
    '''
    Calculates the intersection length or linear attenuation for a sphere in
    parallel beam geometry for 
    
    Parameters
    ----------
    s : array type (x0,y0,r,val)
        Sphere parameters
    ang : float or (nAngs) array_like
        The projection angle(s) in radians
    u : float or (nDets) array_like
        The detector column postion(s)   
        
    Returns
    -------
    sinogram : float or ndarray (nViews,nBins)
        Intersection length
    '''
    ang = np.asarray(ang)
    u   = np.asarray(u)
    
    x0, y0, r = s[0], s[1], s[2]

    # projection angle (nAng, 1)
    phi = (x0*np.sin(ang) - y0*np.cos(ang))
    phi = phi.reshape(ang.shape + (1,)*u.ndim)
    
    u = u.reshape((1,)*ang.ndim + u.shape)

    p = phi + u

    return 2.*np.sqrt((r**2 - p**2).clip(0))


def analytic_sino_circle_fan(s, ang, u, DSO, DSD):
    '''
    Analytic fan-beam sinogram of a circle (flat detector)
    Output shape: (len(ang), len(u))
    '''

    ang = np.asarray(ang)
    u   = np.asarray(u)

    x0, y0, r = s[0], s[1], s[2]

    # fan angle (1, nU)
    gamma = np.arctan(u/DSD)
    gamma = gamma.reshape((1,)*ang.ndim + u.shape)
    
    # parallel offset from fan geometry (1, nU)
    xi = (DSO * np.sin(gamma))
    xi = xi.reshape((1,)*ang.ndim + u.shape)

    # projection angle (nAng, 1)
    #phi = ang[:, None] + gamma    
    phi = ang.reshape(ang.shape + (1,)*u.ndim) + gamma

    # parallel-beam distance evaluated along fan rays
    p = x0*np.sin(phi) - y0*np.cos(phi) + xi

    return 2.0 * np.sqrt((r**2 - p**2).clip(0))


def analytic_sino_sphere_cone(s, ang, u, v, DSO, DSD):
    '''
    Analytic cone-beam sinogram of a sphere (flat panel)

    Output shape:
        (len(ang), len(u), len(v))
    '''

    x0, y0, z0, r = s[0], s[1], s[2], s[3]

    ang = np.asarray(ang)
    u   = np.asarray(u)
    v   = np.asarray(v)

    # Fan angles (1, nU, 1)
    gamma = np.arctan(u / DSD)
    gamma = gamma.reshape((1,)*ang.ndim + u.shape + (1,)*v.ndim)


    # Cone angles (1, 1, nV)
    eta = np.arctan(v / DSD)
    eta = eta.reshape((1,)*ang.ndim + (1,)*u.ndim + v.shape )
   
   
    # Parallel offset from fan geometry
    xi = DSO * np.sin(gamma)          # (1, nU, 1)

    # Projection angle
    phi = ang.reshape(ang.shape + (1,)*u.ndim + (1,)*v.ndim) + gamma  # (nAng, nU, 1)

    # In-plane distance (fan-beam part)
    p_xy = (x0*np.sin(phi) - y0*np.cos(phi) + xi) # (nAng, nU, 1)

    # Out-of-plane distance
    p_z = z0 - DSO * np.tan(eta)       # (1, 1, nV)

    # Total squared perpendicular distance
    d2 = p_xy**2 + p_z**2              # (nAng, nU, nV)

    return 2.0 * np.sqrt((r**2 - d2).clip(0))


def analytic_sino_sphere_cone0(s, ang, u, v, DSO, DSD):
    '''
    Analytic cone-beam sinogram of a sphere (flat panel)

    Output shape:
        (len(ang), len(u), len(v))
    '''

    x0, y0, z0, r = s[0], s[1], s[2], s[3]

    ang = np.asarray(ang)
    u   = np.asarray(u)
    v   = np.asarray(v)

    # Fan angles (1, nU, 1)
    gamma = np.arctan(u / DSD)[None, :, None]

    # Cone angles (1, 1, nV)
    eta = np.arctan(v / DSD)[None, None, :]

    # Parallel offset from fan geometry
    xi = DSO * np.sin(gamma)          # (1, nU, 1)

    # Projection angle
    phi = ang[:, None, None] + gamma  # (nAng, nU, 1)

    # In-plane distance (fan-beam part)
    p_xy = (x0*np.sin(phi) - y0*np.cos(phi) + xi) # (nAng, nU, 1)

    # Out-of-plane distance
    p_z = z0 - DSO * np.tan(eta)       # (1, 1, nV)

    # Total squared perpendicular distance
    d2 = p_xy**2 + p_z**2              # (nAng, nU, nV)

    return 2.0 * np.sqrt((r**2 - d2).clip(0))



def analytic_sino_sphere_gen(src, det, sphere, rho=1.0):
    """
    Vectorized, broadcastable ray-sphere path length.

    src  : (...,3) source positions
    det  : (...,3) detector positions
    sphere: (cx,cy,cz,r)

    Returns: (...,) path length along each ray
    """
    C = np.asarray(sphere[:3])  # Sphere center
    r = sphere[3]               # Sphere radius

    # Vector from source to detector
    v = det - src  

    # Normalize ray direction for numerical stability
    vnorm = np.linalg.norm(v, axis=-1, keepdims=True)
    vhat = v / vnorm

    # Vector from source to sphere center
    w = C - src

    # Projection of w onto ray direction
    t = np.sum(w * vhat, axis=-1)

    # Squared perpendicular distance from ray to sphere center
    d2 = np.sum(w*w, axis=-1) - t*t

    # Initialize projection array
    proj = np.zeros_like(d2)

    # Only rays intersecting the sphere contribute
    mask = d2 < r*r
    proj[mask] = 2 * rho * np.sqrt(r*r - d2[mask])

    return proj



##################################################
#Analytic Sinograms - Line Integration - Polyedron
##################################################

def analytic_sino_polyhedron_cone(planes, ang, u, v, DSO, DSD):    
    src0 = np.array([DSO,0,0])
    det0 = np.array([-(DSD-DSO),0,0])

    det_cnts, src_cnts, u_hats, v_hats = circular_detector_geometry(ang, det0, src0)    
    proj = np.empty([ang.size,u.size,v.size])
 
    for i, (src_cnt,det_cnt,u_hat,v_hat) in enumerate(zip(src_cnts,det_cnts,u_hats,v_hats)):
        dets = detector_grid(det_cnt, u, v, u_hat=u_hat, v_hat=v_hat)
        proj[i,:,:] = analytic_sino_polyhedron_gen(src_cnt, dets, u, v, u_hat, v_hat, planes)

    return proj


def analytic_sino_polyhedron_gen(src, det, u_cnt, v_cnt, u_hat, v_hat, planes):
    """
    Compute line-segment intersection lengths between
    src->trgs rays and a convex cube defined by planes.

    Parameters
    ----------
    trgs : (..., 3) ndarray
        Ray endpoints.
    src : (3,) ndarray
        Common source point.
    planes : (6, 4) ndarray
        Plane coefficients [a,b,c,d].
        Interior must satisfy:
            ax + by + cz + d <= 0

    Returns
    -------
    lengths : (...) ndarray
        Length of segment inside cube.
    """

    src = np.asarray(src, dtype=float)
    planes = np.asarray(planes, dtype=float)

    uv = detector_grid(det, u_cnt, v_cnt, u_hat=u_hat, v_hat=v_hat)
    d = uv - src                     # ray direction
    ray_len = np.linalg.norm(d, axis=-1)

    t_enter = np.zeros(ray_len.shape)
    t_exit = np.ones(ray_len.shape)

    for plane in planes:
        n = plane[:3]
        c = plane[3]

        num = -(src @ n - c)
        den = np.sum(d * n, axis=-1)

        parallel = np.abs(den) < EPS

        # Parallel and outside -> reject
        outside = parallel & ((src @ n + c) > 0)

        t = np.divide(num,den,out=np.zeros_like(den),where=~parallel)

        t_enter = np.where(den < 0,np.maximum(t_enter, t),t_enter)
        t_exit = np.where(den > 0,np.minimum(t_exit, t),t_exit)

        t_enter[outside] = 1
        t_exit[outside] = 0

    valid = (t_exit > t_enter)

    return np.where(valid,(t_exit - t_enter) * ray_len,0.0)




############################################################
#Analytic Sinograms - Volume Integration - Circles / Spheres
############################################################
def analytic_sino_sphere_cone_vol(s, ang, u, v, DSO, DSD):
    
    du = u[1] - u[0]
    dv = v[1] - v[0]
    
    src0 = np.array([DSO,0,0])
    det0 = np.array([-(DSD-DSO),0,0])

    det_cnts, src_cnts, u_hats, v_hats = circular_detector_geometry(ang, det0, src0)    
    proj = np.empty([ang.size,u.size,v.size])
 
    for i, (src_cnt,det_cnt,u_hat,v_hat) in enumerate(zip(src_cnts,det_cnts,u_hats,v_hats)):
        dets = detector_grid(det_cnt, u, v, u_hat=u_hat, v_hat=v_hat)
        proj[i,:,:] = analytic_sino_sphere_gen_vol(src_cnt, dets, u_hat, v_hat,\
                                                   du, dv, s)

    return proj

def analytic_sino_sphere_gen_vol(
        src_centers,      # (...,3) array of source positions with arbitrary leading dimensions
        det_centers,      # (Nu,Nv,3) detector pixel centers
        eu, ev,           # detector basis vectors (3,)
        du, dv,           # pixel size
        sphere,
        src_size=(0,0),   # focal spot size (sy, sz)
        src_nodes=2,      # Gaussian nodes for source/focal spot
        det_nodes=4,      # Gaussian nodes per detector pixel
        rho=1.0
    ):
    """
    Vectorized CT projection for spheres using Gaussian quadrature.

    Supports src_centers of arbitrary shape (...,3) and returns
    sinogram of shape (..., Nu, Nv)
    """
    
    src_centers = np.asarray(src_centers)

    if src_centers.ndim == 1:
        src_centers = src_centers[None, :]
    
    src_shape = src_centers.shape[:-1]  # save leading dimensions
    Nsrc = np.prod(src_shape)           # flatten leading dimensions
    Nu, Nv, _ = det_centers.shape

    # --- Flatten source centers for computation ---
    src_flat = src_centers.reshape(Nsrc, 3)  # shape (Nsrc,3)

    # --- Gaussian quadrature nodes and weights ---
    gu, wu = np.polynomial.legendre.leggauss(det_nodes)
    gv, wv = np.polynomial.legendre.leggauss(det_nodes)
    gs, ws = np.polynomial.legendre.leggauss(src_nodes)

    # --- Detector pixel offsets ---
    U, V = np.meshgrid(gu, gv, indexing='ij')
    det_offsets = 0.5*du*U[...,None]*eu + 0.5*dv*V[...,None]*ev
    det_offsets = det_offsets.reshape(-1,3)  # (det_nodes^2,3)
    det_w = np.outer(wu,wv).ravel()   # detector quadrature weights

    # --- Source/focal spot offsets ---
    SY, SZ = np.meshgrid(gs, gs, indexing='ij')
    src_offsets = np.zeros((src_nodes**2,3))
    src_offsets[:,1] = 0.5*src_size[0]*SY.ravel()  # y offsets
    src_offsets[:,2] = 0.5*src_size[1]*SZ.ravel()  # z offsets
    src_w = np.outer(ws, ws).ravel()               # source quadrature weights

    # --- Flatten detector grid ---
    det_flat = det_centers.reshape(-1,3)  # (Nu*Nv,3)

    # Initialize output
    proj = np.zeros((Nsrc, Nu*Nv))

    # --- Main quadrature loops (over quadrature nodes only, small loops) ---
    for i, doff in enumerate(det_offsets):
        det_batch = det_flat + doff  # apply detector offset
        
        
        for j, soff in enumerate(src_offsets):
            src_batch = src_flat + soff[None,:]  # apply source/focal spot offset

            # Broadcasted ray-sphere intersection
            # returns shape (Nsrc, Nu*Nv)
            p = analytic_sino_sphere_gen(src_batch[:,None,:], det_batch[None,:,:], sphere, rho=rho)
            proj += det_w[i] * src_w[j] * p

    # Normalize by total quadrature weight
    proj /= (np.sum(det_w) * np.sum(src_w))

    # --- Reshape back to original source dimensions ---
    return proj.reshape(*src_shape, Nu, Nv)



############################################################
#Analytic Sinograms - Volume Integration - Polyhedron
############################################################

# Convex hull volume
def convex_polyhedron_volume(vertices):
    if len(vertices) < 4:
        return 0.0
    
    try:
        return ConvexHull(vertices).volume
    except QhullError:
        return 0.0


def analytic_sino_polyhedron_gen_vol(src,det,u_bnd,v_bnd,u_hat,v_hat,planes_polyhedron):
    
    Nu = len(u_bnd) - 1
    Nv = len(v_bnd) - 1
    nfaces = planes_polyhedron.shape[0]

    # Detector boundary coordinates
    uv = detector_grid(det, u_bnd, v_bnd, u_hat=u_hat, v_hat=v_hat)
    u_coord_bot = uv[:, 0, :]
    u_coord_top = uv[:, -1, :]
    v_coord_lft = uv[0, :, :]
    v_coord_rgt = uv[-1, :, :]
    
    # Frustum planes
    ref_u = src + np.array([0,1,0])
    ref_v = src + np.array([0,0,1])
    planes_u = plane_from_pts(src,u_coord_bot,u_coord_top,ref_u)
    planes_v = plane_from_pts(src,v_coord_lft,v_coord_rgt,ref_v)

    planes_uv = plane_from_pts(u_coord_bot[0],u_coord_top[0],u_coord_bot[-1],src)
    
    planes = np.empty([nfaces+5,4],dtype=float)
    planes[5:,:] = planes_polyhedron
    planes[4,:] = planes_uv
    
    proj = np.empty((Nu, Nv))    
    for i in range(Nu):
        planes[0,:] = planes_u[i,:]
        planes[1,:] = -planes_u[i+1,:]
        
        for j in range(Nv):
            planes[2,:] = planes_v[j,:]
            planes[3,:] = -planes_v[j+1,:]
            verts = intersection_vertices(planes)
            
            proj[i,j] = convex_polyhedron_volume(verts)

    return proj


def analytic_sino_polyhedron_cone_vol(planes_polyhedron, ang, u_bnd, v_bnd, DSO, DSD):
    
    
    src0 = np.array([DSO,0,0])
    det0 = np.array([-(DSD-DSO),0,0])

    det_cnts, src_cnts, u_hats, v_hats = circular_detector_geometry(ang, det0, src0)    
    proj = np.empty([ang.size,u_bnd.size-1,v_bnd.size-1])
 
    for i, (src_cnt,det_cnt,u_hat,v_hat) in enumerate(zip(src_cnts,det_cnts,u_hats,v_hats)):
        proj[i,:,:] =  analytic_sino_polyhedron_gen_vol(src_cnt,det_cnt,u_bnd,v_bnd,u_hat,v_hat,planes_polyhedron)


    return proj



##################
#Utility Functions
##################

def detector_grid(det_center, u_arr, v_arr, u_hat=[1.,0.,0.], v_hat=[0.,0.,1.]):
    """
    Calculate detector pixel-center coordinates.

    Parameters
    ----------
    det_center : (3,) array_like
        Detector center position.
    u_arr : (m,) array_like
        Pixel-center offsets along u direction.
    v_arr : (n,) array_like
        Pixel-center offsets along v direction.
    u_hat : (3,) array_like
        Unit vector along detector u axis.
    v_hat : (3,) array_like
        Unit vector along detector v axis.

    Returns
    -------
    coords : (m, n, 3) ndarray
        XYZ coordinates of each detector pixel center.
    """
    det_center = np.asarray(det_center)
    u_arr = np.asarray(u_arr)
    v_arr = np.asarray(v_arr)
    u_hat = np.asarray(u_hat)
    v_hat = np.asarray(v_hat)

    return det_center[None, None, :] + \
             u_arr[:, None, None] * u_hat[None, None, :] + \
             v_arr[None, :, None] * v_hat[None, None, :]


def detector_grid_quadrature(det_center,u_arr,v_arr,du=None,dv=None,det_nodes=(2,2),\
                             u_hat=[1.,0.,0.],v_hat=[0.,0.,1.]):

    det_center = np.asarray(det_center)
    u_arr = np.asarray(u_arr)
    v_arr = np.asarray(v_arr)
    u_hat = np.asarray(u_hat)
    v_hat = np.asarray(v_hat)

    
    if du is None:
        du = u_arr[1] - u_arr[0]

    if dv is None:
        dv = v_arr[1] - v_arr[0]

    gu, wu = np.polynomial.legendre.leggauss(det_nodes[0])
    gv, wv = np.polynomial.legendre.leggauss(det_nodes[1])

    Uq, Vq = np.meshgrid(gu, gv, indexing='ij')

    uq = (0.5 * du * Uq).ravel()
    vq = (0.5 * dv * Vq).ravel()

    wq = np.outer(wu, wv).ravel()

    Nq = len(wq)

    det_positions = np.empty((len(u_arr), len(v_arr), Nq, 3))

    
    det_positions = det_center[None, None, None, :] + \
        + (u_arr[:, None, None] + uq[None, None, :])[..., None] * u_hat + \
        + (v_arr[None, :, None] + vq[None, None, :])[..., None] * v_hat
    

    return det_positions, wq



def circular_detector_geometry(angles, det_center0, src_center0, isocenter=(0.0, 0.0, 0.0)):
    """
    Circular detector trajectory about the z-axis.

    Parameters
    ----------
    angles : (n,) array_like
        Projection angles [rad].
    det_center0 : (3,) array_like
        Detector center at angle=0.
    isocenter : (3,) array_like
        Center of rotation.

    Returns
    -------
    det_center : (n, 3)
    u_hat : (n, 3)
    v_hat : (n, 3)
    """

    angles = np.asarray(angles)
    det_center0 = np.asarray(det_center0, dtype=float)
    src_center0 = np.asarray(src_center0, dtype=float)
    isocenter = np.asarray(isocenter, dtype=float)

    # Detector center relative to isocenter
    r0 = det_center0 - isocenter
    r1 = src_center0 - isocenter

    c = np.cos(angles)
    s = np.sin(angles)

    # Rotate about z-axis
    det_center = np.column_stack([c * r0[0] - s * r0[1],
                                  s * r0[0] + c * r0[1],
                                  np.full_like(c, r0[2]),
                                  ]) + isocenter

    src_center = np.column_stack([c * r1[0] - s * r1[1],
                                  s * r1[0] + c * r1[1],
                                  np.full_like(c, r1[2]),
                                  ]) + isocenter

    # Detector normal points toward isocenter
    n_hat = isocenter - det_center
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)

    # Vertical direction fixed along z
    v_hat = np.tile([0.0, 0.0, 1.0], (len(angles), 1))

    # Horizontal detector direction
    u_hat = np.cross(v_hat, n_hat)
    u_hat /= np.linalg.norm(u_hat, axis=1, keepdims=True)

    return det_center, src_center, u_hat, v_hat


def cube_planes(center, side_lengths):
    center = np.asarray(center, dtype=float)
    sx, sy, sz = side_lengths

    mins = center - np.array([sx, sy, sz]) / 2.
    maxs = center + np.array([sx, sy, sz]) / 2.
    
    planes = np.zeros((6,4))
    planes[0,:] = [1.,0.,0.,maxs[0]]
    planes[1,:] = [-1.,0.,0.,-mins[0]]
    planes[2,:] = [0.,1.,0.,maxs[1]]
    planes[3,:] = [0.,-1.,0.,-mins[1]]
    planes[4,:] = [0.,0.,1.,maxs[2]]
    planes[5,:] = [0.,0.,-1.,-mins[2]]
    
    return planes


def plane_from_pts(p0, p1, p2, reference_point):
    """
    Compute plane coefficients from three points.

    The plane equation is

        nx*x + ny*y + nz*z = d

    and is returned as

        [nx, ny, nz, d]

    Parameters
    ----------
    p0 : (3,) or (N, 3) ndarray
        First point on the plane.
    p1 : (3,) or (N, 3) ndarray
        Second point on the plane.
    p2 : (3,) or (N, 3) ndarray
        Third point on the plane.
    reference_point : (3,) or (N, 3) ndarray
        Point used to orient the plane normal. The returned plane
        is flipped if necessary so that the reference point lies on
        the negative side of the plane.

    Returns
    -------
    plane : (4,) or (N, 4) ndarray
        Plane coefficients [nx, ny, nz, d].

        If a single plane is computed, shape is (4,).

        If multiple planes are computed, shape is (N, 4).

    Notes
    -----
    The normal vector is normalized to unit length.

    The function supports NumPy broadcasting. For example:

        p0.shape == (3,)
        p1.shape == (N, 3)
        p2.shape == (N, 3)

    computes N planes that all share the same p0.
    """

    # Compute (unnormalized) plane normals.
    # Result shape is either (3,) or (N, 3).
    n = np.cross(p1 - p0, p2 - p0)

    # Normalize the normals to unit length.
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    # Compute plane offsets:
    #
    #     n · x = d
    #
    # For batched inputs this performs a row-wise dot product.
    d = np.sum(n * p0, axis=-1)

    # Combine [nx, ny, nz] and d into a single array.
    #
    # d[..., None] converts:
    #   scalar -> (1,)
    #   (N,)   -> (N, 1)
    plane = np.concatenate((n, d[..., None]), axis=-1)

    # Determine which planes need to be flipped so that the
    # reference point lies on the negative side.
    side = np.sum(n * reference_point, axis=-1)
    flip = side > d + EPS

    # Flip planes individually when processing batches.
    plane = np.where(flip[..., None], -plane, plane)

    return plane


def intersection_vertices(planes):
    verts = []

    for idx in combinations(range(len(planes)), 3):
        A = planes[idx,:3]

        if abs(np.linalg.det(A)) < EPS:
            continue

        x = np.linalg.solve(A, planes[idx,3])

        if np.all(planes[:, :3] @ x <= planes[:, 3] + EPS):
            verts.append(x)

    return np.unique(np.round(np.asarray(verts), 8), axis=0)


def unique_vertices(vertices, tol=1e-10):
    """
    Remove duplicate vertices within a Euclidean tolerance.
    """
    vertices = np.asarray(vertices, dtype=float)

    unique = []

    for v in vertices:
        if not any(np.linalg.norm(v - u) < tol for u in unique):
            unique.append(v)

    return np.asarray(unique)

def clipped_sphere_vertices(planes, center, radius):

    v1 = intersection_vertices(planes)

    v2 = plane_plane_sphere_vertices(
        planes,
        center,
        radius
    )

    if len(v1)==0:
        return v2

    if len(v2)==0:
        return v1

    return unique_vertices(
        np.vstack([v1,v2])
    )



def plane_plane_sphere_vertices(planes, center, radius, eps=EPS):

    center = np.asarray(center, dtype=float)

    verts = []

    for i, j in combinations(range(len(planes)), 2):

        n1 = planes[i, :3]
        d1 = planes[i, 3]

        n2 = planes[j, :3]
        d2 = planes[j, 3]

        # line direction
        u = np.cross(n1, n2)

        nu = np.linalg.norm(u)

        # parallel planes
        if nu < eps:
            continue

        # normalize direction
        u = u / nu


        # point on line closest to origin
        A = np.vstack([n1, n2, u])
        b = np.array([d1, d2, 0.0])

        x0 = np.linalg.solve(A, b)


        # line-sphere intersection

        q = x0 - center

        a = np.dot(u, u)
        bb = 2*np.dot(u, q)
        cc = np.dot(q, q) - radius**2

        disc = bb*bb - 4*a*cc

        if disc < -eps:
            continue

        if disc < 0:
            disc = 0

        sqrt_disc = np.sqrt(disc)

        for t in [
            (-bb + sqrt_disc)/(2*a),
            (-bb - sqrt_disc)/(2*a)
        ]:

            x = x0 + t*u

            # must satisfy all clipping planes
            if np.all(
                planes[:,:3] @ x <= planes[:,3] + eps
            ):
                verts.append(x)


    if len(verts)==0:
        return np.empty((0,3))

    return np.asarray(verts)



#########
##TESTING
#########
def sphere_clip_volume0(planes, center, radius,dirs, weights):
    """
    Volume of

        sphere(center,radius) ∩ {x : n·x <= d}

    Parameters
    ----------
    planes : (M,4)
        Plane coefficients [nx,ny,nz,d].

    center : (3,)
        Sphere center.

    radius : float
        Sphere radius.

    dirs : (Nq,3)
        Unit Lebedev directions.

    weights : (Nq,)
        Lebedev weights summing to 4*pi.

    Returns
    -------
    volume : float
    """
    EPS=0

    volume = 0.0

    center = np.asarray(center, dtype=float)

    for w, wt in zip(dirs, weights):

        rmin = 0
        rmax = radius

        valid = True

        for plane in planes:

            n = plane[:3]
            d = plane[3]

            a = np.dot(n, w)
            b = d - np.dot(n, center)

            if abs(a) < EPS:

                if b < 0:
                    valid = False
                    break

            elif a > 0:

                rmax = min(rmax, b / a)

            else:

                rmin = max(rmin, b / a)

            if rmax <= rmin:
                valid = False
                break

        if valid:
            volume += wt * (rmax**3 - rmin**3)

    return volume / 3.0


def sphere_clip_volume1(planes, center, radius, dirs, weights, eps=1e-12):
    """
    Compute the volume of

        sphere(center, radius) ∩ {x : n·x <= d}

    using radial integration over Lebedev directions.

    Parameters
    ----------
    planes : (M,4) array_like
        Plane coefficients [nx, ny, nz, d].

    center : (3,) array_like
        Sphere center.

    radius : float
        Sphere radius.

    dirs : (N,3) array_like
        Unit Lebedev directions.

    weights : (N,) array_like
        Lebedev weights (should sum to 4*pi).

    eps : float, optional
        Numerical tolerance.

    Returns
    -------
    float
        Clipped sphere volume.
    """

    center = np.asarray(center, dtype=float)
    volume = 0.0


    for w, wt in zip(dirs, weights):

        rmin = -np.inf
        rmax = np.inf
        valid = True

        for plane in planes:

            n = plane[:3]
            d = plane[3]

            a = np.dot(n, w)
            b = d - np.dot(n, center)

            print("w:",w,"wt:",wt,"a:",a ,"b:",b, end="")
            # Relative tolerance for detecting parallel rays
            atol = eps * max(1.0, np.linalg.norm(n))

            if abs(a) <= 0:
                # Ray is effectively parallel to plane
                print()

                if b < 0:
                    valid = False
                    print("b<0 - Not valid")
                    break
                continue

            t = b / a
            if a > 0:
                rmax = min(rmax, t)
            else:
                rmin = max(rmin, t)
            print(" t:", t,"rmax:",rmax,"rmin:",rmin)



            # Reject only if interval is definitely empty
            if rmax < rmin:
                print("rmax < rmin - Not valid","rmax:",rmax,"rmin:",rmin, "t:", t)
                valid = False
                break

        if not valid:
            continue

        # Restrict to the sphere
        rmin = max(rmin, 0.0)
        rmax = min(rmax, radius)

        if rmax > rmin :
            print("Add to volume:","rmax:",rmax,"rmin:",rmin,"contribution:",wt * (rmax**3 - rmin**3)/3)
            volume += wt * (rmax**3 - rmin**3)
        else:
            print("rmax < rmin +ATOL - Not valid","rmax:",rmax,"rmin:",rmin)

    return volume / 3.0

def sphere_clip_volume_precomp(A, b, radius, weights):
    """
    Parameters
    ----------
    A : (Nq,M)
        A[k,j] = dirs[k] · normals[j]

    b : (M,)
        b[j] = d_j - normals[j] · center

    radius : float

    weights : (Nq,)

    Returns
    -------
    volume : float
    """

    volume = 0.0

    Nq, M = A.shape

    parallel = 0
    upper = 0
    lower = 0

    for k in range(Nq):

        rmin = 0.0
        rmax = radius
        valid = True

        for j in range(M):

            a = A[k, j]
            
            if abs(a) < EPS:
                parallel += 1
            elif a > 0:
                upper += 1
            else:
                lower += 1
            
            bj = b[j]

            if abs(a) < EPS:
                if bj <= 0.0:
                    valid = False
                    break

            elif a > 0.0:
                t = bj / a

                if t < rmax:
                    rmax = t

            else:
                t = bj / a

                if t > rmin:
                    rmin = t

            if rmax <= rmin:
                valid = False
                break

        if valid:
            volume += weights[k] * (rmax**3 - rmin**3)

    print("Parallel:", parallel,"Upper:", upper,"Lower:", lower)
    return volume / 3.0


"""
def direction_inside(planes, center, w, eps=1e-12):
    for plane in planes:
        a = np.dot(n, w)
        b = d - np.dot(n, center)

        atol = eps

        if abs(a) <= atol:
            if b < -atol:
                return False
            continue

        t = b / a

        if a > 0:
            if t < 0:
                return False

    return True
"""


def analytic_sino_sphere(src,det,u_bnd,v_bnd,u_hat,v_hat,
                         center,radius,rho=1.0):

    
    #Lebedav direcctions and weights
    dirs, weights = lebedev_rule(131)
    dirs = dirs.T
    
    #MNumber of detector elements
    
    Nu = len(u_bnd) - 1
    Nv = len(v_bnd) - 1

    #Detector grid intersections
    uv = detector_grid(det,u_bnd,v_bnd,u_hat=u_hat,v_hat=v_hat)

    #Calculate u planes from the top and bottoms intersection points    
    ref_u = src + np.array([0, 1, 0])
    planes_u = plane_from_pts(src,uv[:,0,:],uv[:,-1,:],ref_u)

    #Calculate v planes from the left and right intersection points    
    ref_v = src + np.array([0, 0, 1])
    planes_v = plane_from_pts(src,uv[0,:,:],uv[-1,:,:],ref_v)

    #Calc detector plane
    planes_uv = plane_from_pts(uv[0, 0,:],uv[0,-1,:],uv[-1,0,:],src)


    planes = np.empty((5, 4), dtype=float)
    planes[4] = planes_uv


    proj = np.zeros((2), dtype=float)

    planes[0] = planes_u[0]
    planes[1] = -planes_u[32]

    planes[2] = planes_v[0]
    planes[3] = -planes_v[16]
    
    print("Hemisphere 1")
    
    
    #vol = sphere_clip_volume1(planes_v[16:17],center,radius,dirs,weights)

    #proj[0] = rho * vol


    planes[2] = planes_v[16]
    planes[3] = -planes_v[32]
    print("Hemisphere 2")
    
    #vol = sphere_clip_volume1(-planes_v[16:17],center,radius,dirs,weights)

    #proj[1] = rho * vol


    print(plane_plane_sphere_vertices(planes_v[16:17], center, radius, eps=EPS))
    """
    proj = np.zeros((Nu, Nv), dtype=float)
    for i in range(Nu):
        planes[0] = planes_u[i]
        planes[1] = -planes_u[i + 1]

        for j in range(Nv):
            planes[2] = planes_v[j]
            planes[3] = -planes_v[j + 1]

            vol = sphere_clip_volume1(planes,center,radius,dirs,weights)

            proj[i, j] = rho * vol
    """

    return proj




def analytic_sino_sphere2(
        src,
        det,
        u_bnd,
        v_bnd,
        u_hat,
        v_hat,
        center,
        radius,
        rho=1):

    dirs, weights = lebedev_rule(131)
    dirs = dirs.T

    
    
    Nu = len(u_bnd) - 1
    Nv = len(v_bnd) - 1

    uv = detector_grid(
        det,
        u_bnd,
        v_bnd,
        u_hat=u_hat,
        v_hat=v_hat
    )

    u_coord_bot = uv[:, 0, :]
    u_coord_top = uv[:, -1, :]

    v_coord_lft = uv[0, :, :]
    v_coord_rgt = uv[-1, :, :]

    ref_u = src + np.array([0.0, 1.0, 0.0])
    ref_v = src + np.array([0.0, 0.0, 1.0])

    planes_u = plane_from_pts(
        src,
        u_coord_bot,
        u_coord_top,
        ref_u
    )
    

    planes_v = plane_from_pts(
        src,
        v_coord_lft,
        v_coord_rgt,
        ref_v
    )

    planes_uv = plane_from_pts(
        u_coord_bot[0],
        u_coord_top[0],
        u_coord_bot[-1],
        src
    )

    proj = np.zeros((Nu, Nv), dtype=float)

    #
    # ---------------------------------------------------------
    # PRECOMPUTE ALL DIR·NORMAL PRODUCTS
    # ---------------------------------------------------------
    #

    Au = dirs @ planes_u[:, :3].T          # (Nq, Nu+1)

    Av = dirs @ planes_v[:, :3].T          # (Nq, Nv+1)

    Auv = dirs @ planes_uv[:3]             # (Nq,)

    #
    # ---------------------------------------------------------
    # PRECOMPUTE ALL b = d - n·center TERMS
    # ---------------------------------------------------------
    #

    bu = planes_u[:, 3] - planes_u[:, :3] @ center
    bv = planes_v[:, 3] - planes_v[:, :3] @ center
    buv = planes_uv[3] - planes_uv[:3] @ center

    #
    # Reusable work arrays
    #

    A = np.empty((len(weights), 5), dtype=float)
    b = np.empty(5, dtype=float)

    for i in range(Nu):

        #
        # u planes
        #

        A[:, 0] = Au[:, i]
        A[:, 1] = -Au[:, i + 1]

        b[0] = bu[i]
        b[1] = -bu[i + 1]

        for j in range(Nv):

            #
            # v planes
            #

            A[:, 2] = Av[:, j]
            A[:, 3] = -Av[:, j + 1]

            b[2] = bv[j]
            b[3] = -bv[j + 1]

            #
            # detector plane
            #

            A[:, 4] = Auv
            b[4] = buv

            #
            # quick reject:
            # sphere completely outside any half-space
            #

            reject = False

            for p in range(5):

                # signed distance to plane
                if -b[p] > radius:
                    reject = True
                    break

            if reject:
                continue

            vol = sphere_clip_volume_precomp(
                A,
                b,
                radius,
                weights
            )

            proj[i, j] = rho * vol

    return proj

