# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:03:35 2026

@author: varga
"""

import numpy as np


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


def analytic_circle_sino_par_2d(s,ang,u):
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




def analytic_circle_sino_fan_2d(s, ang, u, DSO, DSD):
    '''
    Analytic fan-beam sinogram of a circle (flat detector)
    Output shape: (len(ang), len(u))
    '''

    ang = np.asarray(ang)
    u   = np.asarray(u)

    x0, y0, r = s[0], s[1], s[2]

    # fan angle (1, nU)
    #gamma = np.arctan(u/DSD)[None, :]
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
    
    """    
        # Source position along tangent (tangent vector)
    t_x = -np.sin(phi)
    t_y =  np.cos(phi)
    
    x_s = -DSO*np.cos(ang)[:, None] + focal_offset * t_x
    y_s = -DSO*np.sin(ang)[:, None] + focal_offset * t_y
    
    # Parallel-beam distance
    p = (x0 - x_s)*np.sin(phi) - (y0 - y_s)*np.cos(phi) + xi
    """

    return 2.0 * np.sqrt((r**2 - p**2).clip(0))


def analytic_sphere_sino_cone_3d(s, ang, u, v, DSO, DSD):
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


def analytic_sphere_sino_cone_3d0(s, ang, u, v, DSO, DSD):
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



def sphere_projection(src, det, sphere, rho=1.0):
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


def sphere_projection_gauss(
        src_centers,      # (...,3) array of source positions with arbitrary leading dimensions
        det_centers,      # (Nu,Nv,3) detector pixel centers
        eu, ev,           # detector basis vectors (3,)
        du, dv,           # pixel size
        sphere,
        src_size=(0,0),   # focal spot size (sy, sz)
        src_nodes=2,      # Gaussian nodes for source/focal spot
        det_nodes=2,      # Gaussian nodes per detector pixel
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
    det_w = np.outer(wu,wv).ravel()          # detector quadrature weights

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
            p = sphere_projection(src_batch[:,None,:], det_batch[None,:,:], sphere, rho=rho)
            proj += det_w[i] * src_w[j] * p

    # Normalize by total quadrature weight
    proj /= (np.sum(det_w) * np.sum(src_w))

    # --- Reshape back to original source dimensions ---
    return proj.reshape(*src_shape, Nu, Nv)







