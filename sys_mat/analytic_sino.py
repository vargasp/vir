# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:03:35 2026

@author: varga
"""

import numpy as np


def sphere_phantom_upsample(s,nx,ny,nz, upsample=1):
    
    x0_u, y0_u, z0_u, r_u = s[0]*upsample, s[1]*upsample, s[2]*upsample, s[3]*upsample
    nx_u, ny_u, nz_u = nx*upsample, ny*upsample, nz*upsample
    
    
    x,y,z = np.indices((nx_u,ny_u,nz_u)) + 0.5
    

    a = (x - nx_u/2. - x0_u)**2 + (y - ny_u/2. - y0_u)**2 + (z - nz_u/2. - z0_u)**2 < r_u**2
    #return a*1

    a = a*1
    shp = (nx,ny,nz)
    
    factors = (np.array(a.shape)/np.asanyarray(shp)).astype(int)
    sh = np.column_stack([a.shape//factors, factors]).ravel()
    return a.reshape(sh).mean(tuple(range(1, 2*a.ndim, 2)))
    

    

def sphere_phantom_approx(s, nx, ny, nz):

    x0, y0, z0, r = s

    x = np.arange(nx)[:,None,None] + 0.5 - nx/2 - x0
    y = np.arange(ny)[None,:,None] + 0.5 - ny/2 - y0
    z = np.arange(nz)[None,None,:] + 0.5 - nz/2 - z0

    d = np.sqrt(x*x + y*y + z*z) - r

    # voxel diagonal gives transition width
    w = np.sqrt(3)/2

    occ = np.clip(0.5 - d/w, 0, 1)

    return occ.astype(np.float32)


def sphere_cube_volume(x0,x1,y0,y1,z0,z1,r,ns=8):
    """
    Exact volume via high-order Gaussian quadrature.
    ns = quadrature order (8–12 gives machine precision)
    """

    # Gauss-Legendre nodes
    xs, ws = np.polynomial.legendre.leggauss(ns)

    # map to interval
    xs = 0.5*(xs+1)*(x1-x0)+x0
    wx = ws*(x1-x0)/2

    ys = 0.5*(xs*0 + np.polynomial.legendre.leggauss(ns)[0]+1)*(y1-y0)+y0
    wy = np.polynomial.legendre.leggauss(ns)[1]*(y1-y0)/2

    vol = 0.0

    for i,x in enumerate(xs):
        for j,y in enumerate(ys):

            d2 = x*x + y*y
            if d2 >= r*r:
                continue

            z = np.sqrt(r*r - d2)

            zlo = max(z0, -z)
            zhi = min(z1,  z)

            if zhi > zlo:
                vol += wx[i]*wy[j]*(zhi-zlo)

    return vol


def sphere_phantom_exact(s, nx, ny, nz):

    x0,y0,z0,r = s

    out = np.zeros((nx,ny,nz),dtype=float)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                xlo = i - nx/2 - x0
                xhi = xlo + 1

                ylo = j - ny/2 - y0
                yhi = ylo + 1

                zlo = k - nz/2 - z0
                zhi = zlo + 1

                out[i,j,k] = sphere_cube_volume(
                    xlo,xhi,ylo,yhi,zlo,zhi,r
                )

    return out

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







