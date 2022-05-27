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


def AnalyticSinoParSphere(S,view,xi):
    '''
    Calculates the intersection length or linear attenuation for a sphere in
    parallel beam geometry for 
    
    Parameters
    ----------
    S : array type (x0,y0,r,val)
        Sphere parameters
    view : float or (nViews) array_like
        The projection angle(s) in radians
    xi : float or (nDets) array_like
        The detector column postion(s)   
        
    Returns
    -------
    sinogram : float or ndarray (nViews,nBins)
        Intersection length
    '''

    u = np.add.outer(np.linalg.norm(S[:2])*np.cos(np.arctan2(S[0], S[1]) - view),xi)

    return 2.*np.sqrt((S[2]**2 - u**2).clip(0))


def AnalyticSinoFanSphere(S,view,Bins,src_iso,offset=0.0):

    xi   = src_iso*np.sin(Bins) + offset*np.cos(Bins)

    s  = np.sqrt(S[0]**2 + S[1]**2)
    delta = np.arctan2(S[0], S[1])

                
    phi  = np.add.outer(view, Bins)
    f    = S[2]**2 * (np.cos(phi)**2 + np.sin(phi)**2)
    
    u    = xi + s*np.cos(delta - phi)

    weight = (2.*S[2]**2/f)*np.sqrt((f - u**2).clip(0) )

    return weight


def AnalyticSinoParSphere2(Sphere,Views,Rows,Cols):
    '''
    Calculates the intersection length for a sphere in
    parallel beam geometry for 
    
    Parameters
    ----------
    S : array type (x0,y0,z0,r)
        Sphere parameters
    Views : float or (nViews) array_like
        The projection angle(s) in radians
    Rows : float or (nRows) array_like
        The detector row postion(s)
    Cols : float or (nCols) array_like
        The detector column postion(s)  
        
    Returns
    -------
    sinogram : float or ndarray (nViews,nRows,nCols)
        Intersection length 
    '''
    Views = np.array(Views)
    Cols = np.array(Cols)
    Rows = np.array(Rows)[np.newaxis]

    if Rows.ndim == 1:
        Rows = Rows[np.newaxis]        

    #Calculates planes at the row z postiions 
    P = np.hstack([np.tile([0,0,1],(Rows.size,1)), -1*Rows.T])

    #Calculates the circle parameters at the row z postiions 
    C = SpherePlaneIntersection(P,Sphere)
    
    if C.ndim == 1:
        C = C[np.newaxis]

    #Calculates the intersection distances
    #u = np.add.outer(np.linalg.norm(C[:,:2],axis=1)* \
    #        np.cos(np.subtract.outer(np.arctan2(C[:,0], C[:,1]),Views).T),Rows)


    u = np.add.outer(np.linalg.norm(C[:,:2],axis=1)* \
            np.cos(np.subtract.outer(Views,np.arctan2(C[:,0], C[:,1]))),Cols)


    if Cols.ndim == 1:
        return 2.*np.sqrt((C[:,3][np.newaxis].T**2 - u**2).clip(0))
    else:
        return 2.*np.sqrt((C[:,3]**2 - u**2).clip(0))
    

def AnalyticSinoParEllipse(S,view,xi):
    '''
    !!!NOT FULLY TESTED!!!
    
    
    Calculates the intersection length or linear attenuation for an ellipse in
    parallel beam geometry for 
    
    Parameters
    ----------
    E : array type (x0,y0,xA,yB,angle, val)
        Ellipse parameters (angle in radians)
    view : float or (nViews) array_like
        The projection angle(s) in radians
    xi : float or (nDets) array_like
        The detector column postion(s)
    value {‘atten’, ‘length’}, optional, default: ‘atten’
        Whether to return intersection lengths or linear attenuation    
        
    Returns
    -------
    sinogram : float or ndarray (nViews,nBins)
        Intersection length or linear attenuation
    '''
    

    
    view = np.array(view)
    xi = np.array(xi)

    eA = S[2]
    eB = S[3]
    eT = S[4]
    eV = S[5]

                
    f = eA**2 * np.cos(view - eT)**2 + eB**2 * np.sin(view - eT)**2
    f = np.tile(f,[xi.size,1]).T
    u = np.add.outer(np.linalg.norm(S[:2])*np.cos(np.arctan2(S[0], S[1]) - view),xi)
    weight = (2.*eA*eB*eV/f)*np.sqrt((f - u**2).clip(0) )
    print(f.shape, u.shape, (f-u).shape, weight.shape)

    return weight


def AnalyticSino(S,view,Cols,src_iso,offset=0.0):
    
    eX = S[0]
    eY = S[1]
    eA = S[2]
    eB = S[2]
    eT = 0.0
    eV = 1.0

    xi   = src_iso*np.sin(Cols) + offset*np.cos(Cols)

    s  = np.sqrt(eX**2 + eY**2)
    delta = np.arctan2(eX, eY)
    term1 = 2.*eA*eB*eV
                
    phi  = view + Cols
    f    = eA**2 * np.cos(phi - eT)**2 + eB**2 * np.sin(phi - eT)**2
    u    = xi + s*np.cos(delta - phi)
    weight = (term1/f)*np.sqrt((f - u**2).clip(0) )

    return weight
    

def sphere_int_tot(X,Y,x0=0,y0=0,r=10):
    f = lambda z, y, x: 1
    gfun = lambda x: -np.sqrt(r**2 - x**2) #The lower boundary curve in y
    hfun = lambda x: np.sqrt(r**2 - x**2) #The upper boundary curve in y 
    qfun = lambda x, y: -np.sqrt(r**2 - x**2 - y**2) #The lower boundary surface in z
    rfun = lambda x, y: np.sqrt(r**2 - x**2 - y**2) #The upper boundary surface in z
    
    return tplquad(f, -r, r, gfun, hfun, qfun, rfun)[0]


def sphere_int_proj1d(X,Y,x0=0,y0=0,r=10):
    V1d = np.zeros(X.size-1)

    f = lambda z, y, x: 1
    gfun = lambda x: -np.sqrt(r**2 - x**2) #The lower boundary curve in y
    hfun = lambda x: np.sqrt(r**2 - x**2) #The upper boundary curve in y 
    qfun = lambda x, y: -np.sqrt(r**2 - x**2 - y**2) #The lower boundary surface in z
    rfun = lambda x, y: np.sqrt(r**2 - x**2 - y**2) #The upper boundary surface in z
    for i, x in enumerate(X[:-1]):
        if np.min([np.abs(x), np.abs(x+1)]) < r:
            V1d[i] = tplquad(f, x, x+1, gfun, hfun, qfun, rfun)[0]
    
    return V1d


def sphere_int_proj2d(X,Y,x0=0,y0=0,r=10):
    """

    Parameters
    ----------
    X : TYPE
        Intersecting lines in the X dimension
    Y : TYPE
        DESCRIPTION.
    x0 : TYPE, optional
        Intersecting lines in the Y dimension
    y0 : TYPE, optional
        DESCRIPTION. The default is 0.
    r : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    print(x0,y0,r)
    V_array = np.zeros([X.size-1,Y.size-1])
    

    f_xyz = lambda z, y, x: 1
    f_yxz = lambda z, x, y: 1
    gfun_y = lambda y: -np.sqrt(r**2 - (y-y0)**2) + x0 #The lower boundary curve in x
    hfun_y = lambda y: np.sqrt(r**2 - (y-y0)**2) + x0 #The upper boundary curve in x 
    gfun_x = lambda x: -np.sqrt(r**2 - (x-x0)**2) + y0 #The lower boundary curve in y
    hfun_x = lambda x: np.sqrt(r**2 - (x-x0)**2) + y0 #The upper boundary curve in y 
    qfun_xy = lambda x, y: -np.sqrt(r**2 - (x-x0)**2 - (y-y0)**2) #The lower boundary surface in z
    rfun_xy = lambda x, y: np.sqrt(r**2 - (x-x0)**2 - (y-y0)**2) #The upper boundary surface in z
    qfun_yx = lambda y, x: -np.sqrt(r**2 - (x-x0)**2 - (y-y0)**2) #The lower boundary surface in z
    rfun_yx = lambda y, x: np.sqrt(r**2 - (x-x0)**2 - (y-y0)**2) #The upper boundary surface in z


    #Extent of the sphere at intersection lines
    with np.errstate(invalid='ignore'):
        XS_l = gfun_y(Y)   #Lower boundary at the Y intersecting lines
        XS_u = hfun_y(Y)   #Upper boundary at the Y intersecting lines
        
    #Limits if integration grom intersections
    XP_l = X[:-1]
    XP_u = X[1:]
    YP_l = Y[:-1]
    YP_u = Y[1:]

    #Boolean flag if line intersection is withhin the projected sphere
    XLYL = (r**2 - np.add.outer((XP_l-x0)**2,(YP_l-y0)**2)) > 0
    XLYU = (r**2 - np.add.outer((XP_l-x0)**2,(YP_u-y0)**2)) > 0
    XUYL = (r**2 - np.add.outer((XP_u-x0)**2,(YP_l-y0)**2)) > 0
    XUYU = (r**2 - np.add.outer((XP_u-x0)**2,(YP_u-y0)**2)) > 0

    #Number of pixel vertices within the projected sphere
    NZ = np.count_nonzero(np.dstack([XLYL,XLYU,XUYL,XUYU]),axis=2)
    #return(NZ)

    #Loops through the pixels an calcuates the projected volume by numerical integration
    for i, (xp_l,xp_u) in enumerate(zip(XP_l,XP_u)):
        xps = (xp_l,xp_u)
        
        for j, (yp_l,yp_u) in enumerate(zip(YP_l,YP_u)):
            xp_l, xp_u = xps
            
            if NZ[i,j] == 0:
                V = 0.0

            elif NZ[i,j] == 4:
                V = tplquad(f_xyz, xp_l, xp_u, yp_l, yp_u, qfun_xy, rfun_xy)[0]
                
            elif NZ[i,j] == 2 or NZ[i,j] == 3:
                #Left vertices
                if (XLYL[i,j] and XLYU[i,j]):
                    V = tplquad(f_yxz, yp_l, yp_u, xp_l, hfun_y, qfun_yx, rfun_yx)[0]
                #Right vertices
                elif (XUYL[i,j] and XUYU[i,j]):
                    V = tplquad(f_yxz, yp_l, yp_u, gfun_y, xp_u, qfun_yx, rfun_yx)[0]
                #Lower vertices
                elif (XLYL[i,j] and XUYL[i,j]):
                    V = tplquad(f_xyz, xp_l, xp_u, yp_l, hfun_x, qfun_xy, rfun_xy)[0]
                #Upper vertices
                elif (XLYU[i,j] and XUYU[i,j]):
                    V = tplquad(f_xyz, xp_l, xp_u, gfun_x, yp_u, qfun_xy, rfun_xy)[0]
                else:
                    print("Logic Error")
        
            elif NZ[i,j] == 1:
                if XLYL[i,j]:
                    V = tplquad(f_xyz, xp_l, XS_u[j], yp_l, hfun_x, qfun_xy, rfun_xy)[0]        
                    #print("1. ",V)
                elif XLYU[i,j]:
                    V = tplquad(f_xyz, xp_l, XS_u[j+1], gfun_x, yp_u, qfun_xy, rfun_xy)[0]        
                    #print("2. ",V)
                elif XUYL[i,j]:
                    V = tplquad(f_xyz, XS_l[j], xp_u, yp_l, hfun_x, qfun_xy, rfun_xy)[0]        
                    #print("3. ",V)
                elif XUYU[i,j]:
                    V = tplquad(f_xyz, XS_l[j+1], xp_u, gfun_x, yp_u, qfun_xy, rfun_xy)[0]        
                    #print("4. ",V)

                else:
                    print("Logic Error")

            else:
                print("Logic Error")
                
            V_array[i,j] = V

    #Correct 3 NZ
    idx = np.where(NZ == 3)
    for c, (i,j) in enumerate(zip(idx[0],idx[1])):
        if XP_l[i] < x0:
            V_array[i,j] = V_array[i,j] - np.sum(V_array[:i,j])
        else:
            V_array[i,j] = V_array[i,j] - np.sum(V_array[(i+1):,j])

    #Checks for spheres within a pixel
    if np.all(V_array == 0.0):
        i = np.where(X > x0)[0][0]
        j = np.where(Y > y0)[0][0]

        if i == X.size-1:
            return V_array
        elif j == Y.size-1:
            return V_array
        else:
            V_array[i,j] = 4./3*np.pi*r**3       
            return V_array
    else:        
        return V_array
