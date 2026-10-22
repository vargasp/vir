#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:32:03 2021

@author: vargasp
"""
import numpy as np


def plane3pts_std(p0,p1,p2):
    """
    Returns the standard form of a plane equation from 3 points
    
    Parameters
    ----------
    p0 : (3) array_like
        Coordinates of the 1st point (x,y,z)
    p1 : (3) array_like
        Coordinates of the 2nd point (x,y,z)
    p2 : (3) array_like
        Coordinates of the 3rd point (x,y,z)

    Returns
    -------
    (4) numpy ndarray 
        The coefficients of the standard form of the plane equation (A,B,C,D)
        from Ax + Bx + Cx = D
    """
    
    #Converts and assigns paramters to approprate data types
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    
    normal = np.cross(p1 - p0, p2 - p0)

    return np.append(normal,p0.dot(normal))


def line2pts_parm(p0,p1):
    """
    Returns the parametric equations of a line from 2 points
    
    Parameters
    ----------
    p0 : (3) or (nPoints,3) array_like
        Coordinates of the 1st point (x,y,z). If nPoints must equal p1 or 
        be equal to one.
    p1 : (3) or (nPoints,3) array_like
        Coordinates of the 2nd point (x,y,z)

    Returns
    -------
    (3,2) or (nPoints,3,2) numpy ndarray 
        The coefficients of the parametric equations of a line
        ((x0,A),(y0,B),(z0,C)) from x = x0 + At, y = y0 + Bt, and z = z0 + Ct
    """
    
    #Converts and assigns paramters to approprate data types
    p0 = np.array(p0)
    p1 = np.array(p1)

    return np.stack((p1,p0-p1),axis=-1)
    
def line_plane_int(pLine,plane):
    """
    Returns the point(s) of intersection between a line(s) and a plane
    
    Parameters
    ----------
    pLine : (3,2) or (nPoints,3,2) array_like
        The coefficients of the parametric equations of a line
        ((x0,A),(y0,B),(z0,C)) from x = x0 + At, y = y0 + Bt, and z = z0 + Ct
    plane : (4) array_like
        Coordinates of the 2nd point (x,y,z)

    Returns
    -------
    The coefficients of the standard form of the plane equation (A,B,C,D)
        from Ax + Bx + Cx = D
    """
    
    #Converts and assigns paramters to approprate data types
    pLine = np.array(pLine,dtype=float)
    if pLine.ndim == 2:
        pLine = pLine[np.newaxis,:,:]
    
    t = (plane[3] - np.sum(pLine[:,:,0]*plane[0:3],axis=1))/np.sum(pLine[:,:,1]*plane[0:3],axis=1)
    t = (plane[3] - np.dot(pLine[:,:,0],plane[0:3]))/np.dot(pLine[:,:,1],plane[0:3])

    return np.squeeze(pLine[:,:,0] + t[:,np.newaxis]*pLine[:,:,1])

    