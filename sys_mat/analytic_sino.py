# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:03:35 2026

@author: varga
"""

import numpy as np


def phantom(s,nx,ny,nz, upsample=1):
    
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



