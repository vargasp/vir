#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:17:49 2020

@author: vargasp
"""

import numpy as np
from scipy.integrate import tplquad

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def broadcast_pts(X,Y,Z):
    """
    Creates an array of 3 coordinate points based on nD arrays of X, Y, X.
    Coordinates are broadcasted to create all combinations 

    Parameters
    ----------
    X : scalar or array_like
        The coordinate(s) of the X positions
    Y : scalar or array_like
        The coordinate(s) of the Y positions
    Z : scalar or array_like
        The coordinate(s) of the Z positions

    Returns
    -------
    (3) or (...,3) numpy ndarray
        The nD array of the combined coordinate points        
    """

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    #Concatentates the broadcasted shape 
    s = X.shape + Y.shape + Z.shape

    #Calculates the axis dimensions to expand
    x_s = tuple(range(X.ndim,len(s)))
    y_s = tuple(range(0,X.ndim)) + tuple(range(len(s) - Z.ndim, len(s)))
    z_s = tuple(range(0,len(s)-Z.ndim))
    
    #Expands the dimensions to meet broadcasting casting rules
    X = np.expand_dims(X,x_s)
    Y = np.expand_dims(Y,y_s)
    Z = np.expand_dims(Z,z_s)
    
    #Creates the array combined array of all points
    return np.stack([np.broadcast_to(X,s), np.broadcast_to(Y,s),np.broadcast_to(Z,s)],axis=-1)


def pts_dist(Pts):
    """
    Calculates the ecludian distance between an array of points coefficients of a line from 2 points 

    Parameters
    ----------
    Pts : (2,3) or (...,2,3) numpy ndarray
        The coordinates of the 2 points
   
    Returns
    -------
    float or (...) numpy ndarray
        The returned distance is coefficients of the parametric lines 
    """
    
    return np.linalg.norm(Pts[...,0,:] - Pts[...,1,:], axis=-1)


def parametric_line(p0,p1):
    """
    Calculates the parametric coefficients of a line from 2 points 

    Parameters
    ----------
    p0 : (3) or (...,3) array_like
        The coordinates of the 1st position. ... shape must equal target or 
        be equal to one.
    p1 : (3) or (...,3) array_like
        The coordinates of the target position. (p1 may have multiple 
        postions and p0 1 postiion)

    Returns
    -------
    (2,3) or (...,2,3) numpy ndarray
        the returned array is coefficients of the parametric lines 
    """
    
    p0 = np.array(p0)
    p1 = np.array(p1)
    
    return np.stack([np.broadcast_to(p0, p1.shape), p1 - p0], -2)


def plane_params(p0,p1,p2):
    """
    Calculates the coeficents [A,B,C,D] of a plane in the general equation form
    from 3 points
        
    Parameters
    ----------
    p0 : (3) array_like
        The coordinates of the 1st point
    p1 : (3) array_like
        The coordinates of the 2nd point
    p2 : (3) array_like
        The coordinates of the 3rd point

    Returns
    -------
    P : (4) array_like
        The coefficients [A,B,C,D] of a plane in the general equation form:
    """
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    normal = np.cross(p1 - p0, p2 - p0)

    return np.append(normal, p0.dot(normal))


def plane_line_inter(L,P):
    """
    Calculates the intersection point(s) of a line and plane 
    intersection 
        
    Parameters
    ----------
    P : (4) array_like
        The coefficients [A,B,C,D] of a plane in the general equation form:
        Ax + By + Cz = D 
    L : (2,3) or (...,2,3) numpy ndarray
        The coefficients of the parametric lines
 
    Returns
    -------
    Pts : (3) or (...,3) array_like
        The coordinates of the intersection points. If the line lies within the
        plane inf for the coordinates are returned. If the line lies parallel
        to, but no in the plane nan for the coordinates are returned. 
    """
    
    #Calculates the dot product of the normal plane vector with the line 
    #position and direction vectors
    nv0 = np.dot(L[...,0,:], P[:3])
    nv1 = np.dot(L[...,1,:], P[:3])

    #Calculates intersection coordinates
    with np.errstate(all='ignore'):
        Pts = L[...,0,:] + L[...,1,:]*((P[3] - nv0)/nv1)[...,np.newaxis]

    #Conforms IEEE 754 Standards
    Pts[np.isneginf(Pts)] = np.inf

    return Pts


def sphere_line_inter(L,S):
    """
    Calculates the intersection point(s) of a line and sphere 
    intersection 
        
    Parameters
    ----------
    S : (4) array_like
        The coefficients [a,b,c,r] of a sphere in the general equation form:
        (x-a)^2 + (y-b)^2 + (z-c)^2 = r^2 
    L : (2,3) or (...,2,3) numpy ndarray
        The coefficients of the parametric lines
 
    Returns
    -------
    Pts : (2,3) or (...,2,3) array_like
        The coordinates of the intersection points. If the line lies within the
        plane inf for the coordinates are returned. If the line lies parallel
        to, but no in the plane nan for the coordinates are returned. 
    """

    L = np.array(L)
    S = np.array(S)
    
    #Sphere center line point difference
    dSL = S[:3]-L[...,0,:]
    
    #Calculates coefs for the quadratic equation
    qe_a = L[...,1,0]**2 + L[...,1,1]**2 + L[...,1,2]**2
    qe_b = -2*(L[...,1,0]*dSL[...,0] + L[...,1,1]*dSL[...,1] + L[...,1,2]*dSL[...,2])
    qe_c = dSL[...,0]**2 + dSL[...,1]**2 + dSL[...,2]**2  - S[...,3]**2

    #Calculatsion roots (np.roots cannot accepts array data)
    with np.errstate(invalid='ignore'):
        term = np.sqrt(qe_b**2 - 4*qe_a*qe_c) 
    roots = np.stack(((-qe_b + term)/2.0/qe_a,(-qe_b - term)/2.0/qe_a),axis=-1)

    return roots[...,np.newaxis]*L[...,np.newaxis,1,:] + L[...,np.newaxis,0,:]


def sphere_plane_inter(P,S):
    """
    Calculates the parameters of a circle defined by the intersection of a
    sphere and plane(s)
        
    Parameters
    ----------
    P : (4) or (nPlanes,4) array_like
        The coeficents [A,B,C,D] of a plane in the general equation form:
        Ax + By + Cz = D 
    S : (4) array_like
        The center of the sphere and radius (x0,y0,z0,r)
 
    Returns
    -------
    C : (4) or (nPlanes,4) array_like
        The circle parameters (x0,y0,z0,r) if there is no intersection returns
        the point the sphere is tangent to the plane an r = 0
    """
    
    P = np.array(P)
    S = np.array(S)
    
    if P.ndim == 1:
        P = P[np.newaxis]
    
    Pt = np.sum(P[:,:3]*S[:3],axis=1)+P[:,3]
    N = np.sum(P[:,:3]**2,axis=1)

    d = np.abs(Pt) / np.sqrt(N)

    with np.errstate(invalid='ignore'):
        r = np.sqrt(S[3]**2 - d**2)

    r = np.nan_to_num(r) 
    
    return np.squeeze(np.hstack((S[:3] -P[:,:3]*(Pt/N)[:,np.newaxis], r[np.newaxis].T)))


def sphere_plane_inter_dist(P,S):
    """
    Calculates the radius and distance of a circle from a sphere and plane
    intersection 
        
    Parameters
    ----------
    P : (4) array_like
        The coeficents [A,B,C,D] of a plane in the general equation form:
        Ax + By + Cz = D 
    S : (4) array_like
        The center of the sphere and radius (x0,y0,z0,r)
 
    Returns
    -------
    r : float
        The radius of the intersected circle
    d : float
        The length of the normal vector from the plane to the center of the
        intersected circle
    """
    
    P = np.array(P)
    S = np.array(S)
    
    d = np.abs(np.sum(P[:3]*S[:3])+P[3]) / np.linalg.norm(P[:3])
    
    with np.errstate(invalid='ignore'):
        return (np.sqrt(S[3]**2 - d**(2)), d)


def plot_plane(P):

    xx, yy = np.meshgrid(range(10), range(10))

    z = (-P[0]*xx - P[1]*yy - P[3])  / P[2]

    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z)
    plt3d.set_xlabel('X Axis')
    plt3d.set_ylabel('Y Axis')
    plt3d.set_zlabel('Z Axis')
    plt.show()
    

