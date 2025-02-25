#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:33:06 2025

@author: pvargas21
"""

import numpy as np

import vir
import vir.affine_transforms as af


def forward_project_phantom_misalign(phantom, Views, \
                                     trans_X=0.0,trans_Y=0.0,trans_Z=0.0,
                                     angX_Z=0.0, angY_Z=0.0, center_Z=None,\
                                     angX_A=0.0, angY_A=0.0, center_A=None):
    """
    Creates a sinogram from a 3d phatom with a precession motion artifact. 

    Parameters
    ----------
    phantom : (nX, nY, nZ) np.array
        A 3d phantom.
    Views : (nViews) np.array
        The array of angles of rotation.
    angX : float
        The initial rotation angle in the X dimension.
    angY : float
        The initial rotation angle in the Y dimension.
    center : (3) array_like, optional
        The center of rotation location. The default is "None" which
        corresponds to the center of the image.

    Returns
    -------
    sino : (nAngs, nY, nZ) np.array
        The forward projected sinogram with a precession artifact.
    """

    nX, nY, nZ = phantom.shape

    if center_A is None:
        center_A = (nX/2.-0.5, nY/2.-0.5, nZ/2.-0.5)

    if center_Z is None:
        center_Z = (nX/2.-0.5, nY/2.-0.5, nZ/2.-0.5)

    
    sino = np.zeros([Views.size,nZ,nX], dtype=np.float32)
    coords = af.coords_array((nX,nY,nZ), ones=True)

    #Misalign phantom by translating axis of rotation with respect to z-axis
    T = af.transMat((trans_X,trans_Y,trans_Z))

    #Misalign phantom by rotating axis of rotation with respect to z-axis
    R_Z = af.rotateMat((angX_Z,angY_Z,0), center=center_Z, seq='XYZ')

    #Loop through views
    for i, view in enumerate(Views):

        #Rotate for tomographic projection and add precession misalignment  
        R_A = af.rotateMat((angX_A,angY_A,view), center=center_A, seq='XYZ')

        #Apply transforms and project view
        RRTC = np.round(R_A @ R_Z @ T @ coords,6)
        sino[i,:,:] = af.coords_transform(phantom, RRTC).sum(axis=1).T
    
    return sino    


def add_detector_tilt_shift(sino3d, r, x):
    
    nAngs, nRows, nCols = sino3d.shape
    
    coords = af.coords_array((nAngs,nRows,nCols), ones=True)

    center = np.array([1.0,nRows,nCols])/2.0 - 0.5
    T, R = proj_orient_TM(r, x, center)
    RTC = (np.linalg.inv(R @ T) @ coords)

    return af.coords_transform(sino3d, np.round(RTC,6))


def proj_orient_TM(r, x, center):
    T = af.transMat((0,0,x))
    R = af.rotateMat((r,0,0), center=center)
    return T, R


def precesion_TM(ang, phi, theta, center):
    r = phi*np.cos(ang + theta)
    z = np.sin(np.pi/2 - phi)
    h_xy = np.cos(np.pi/2 - phi) 
    
    x = np.cos(ang + theta)*h_xy
    s = np.sqrt(x**2 + z**2)

    S = af.scaleMat((1,1/s,1))
    R = af.rotateMat((r,0,0), center=center)
    return S, R


def calib_det_orient(sino3d, Angs, ang, r, x):
    nViews, nRows, nCols = sino3d.shape
    
    coords = af.coords_array((1,nRows,nCols), ones=True)
    coords[:,:,0,:] = np.interp(ang,Angs,np.arange(nViews))
    
    center = np.array([1.0,nRows,nCols])/2.0 - 0.5
    T, R = proj_orient_TM(-r, -x, center)
    
    TRC = np.linalg.inv(T @ R) @ coords
    return np.squeeze(af.coords_transform(sino3d, TRC))


def calib_precesion(sino3d, Angs, ang, phi, theta, center):
    nViews, nRows, nCols = sino3d.shape

    coords = af.coords_array((1,nRows,nCols), ones=True)
    coords[:,:,0,:] = np.interp(ang,Angs,np.arange(nViews))

    S, R = precesion_TM(ang, phi, theta, center)

    SRC = np.linalg.inv(S @ R) @ coords
    return np.squeeze(af.coords_transform(sino3d, SRC))


def calib_proj_orient_view(sino3d, Views, view, z,\
                           transX=0.0, rZ=0.0, cenZ_y=None,\
                           phi=0.0, theta=0.0, cenA_y=None, \
                           transD=0.0, rD=0.0):

    nViews, nRows, nCols = sino3d.shape

    #Creats coordinate arrays
    coords = af.coords_array((1,nRows,nCols), ones=True)
    coords[:,:,0,:] = np.interp(view,Views,np.arange(nViews))


    #Center of of rotation transforms
    if cenZ_y is None: cenZ_y = nRows/2.-0.5 - z
    else: cenZ_y -= z
        
    center_Z = np.array((0.0, cenZ_y, nCols/2.-0.5))
    T_Z, R_Z = proj_orient_TM(-rZ, -transX, center_Z)

    
    #Precession Transforms
    if cenA_y is None: cenA_y = nRows/2.-0.5
    else: cenA_y -= z    

    center_A = np.array((0.0, cenA_y, nCols/2.-0.5))
    S_A, R_A = precesion_TM(view, phi, theta, center_A)
    
    
    #Detector Transforms     
    center_D=(0.0,nRows/2-0.5,nCols/2-0.5)
    T_D, R_D = proj_orient_TM(rD, transD, center_D)  


    #Apply Transforms    
    SRTR = np.linalg.inv(T_D @ R_D @ S_A @ R_A @ T_Z @ R_Z) @ coords
    return np.squeeze(af.coords_transform(sino3d, SRTR))


def calib_proj_orient(sino3d, Views, transX=0.0, rZ=0.0, cenZ_y=None,\
                      phi=0.0, theta=0.0, cenA_y=None, transD=0.0, rD=0.0,\
                      pitch=0):
    nViews, nRows, nCols = sino3d.shape
    
    if cenA_y is None:
        cenA_y = nRows/2.-0.5
    center_A = np.array((0.0, cenA_y, nCols/2.-0.5))

    if cenZ_y is None:
        cenZ_y = nRows/2.-0.5
    center_Z = np.array((0.0, cenZ_y, nCols/2.-0.5))
    
    center_D=(0.0,nRows/2-0.5,nCols/2-0.5)
    T_D, R_D = proj_orient_TM(rD, transD, center_D)  
    
    
    dView = (Views[-1] - Views[0])/(nViews-1)
    dZ = pitch*nRows*dView/(np.pi*2)
    c = 128 - nRows/2 +1
    Z = vir.censpace(nViews,c=c,d=dZ)
    
    coords = af.coords_array((1,nRows,nCols), ones=True)
    #coords[:,:,0,:] = np.interp(ang,Angs,np.arange(nViews))
    

    for i in range(nViews):
        coords[:,:,0,:] = i
        view = Views[i]

        #Center of of rotation transforms
        center_Z[1] = cenZ_y-Z[i]
        T_Z, R_Z = proj_orient_TM(-rZ, -transX, center_Z)

        #Precession Transforms
        center_A[1] = cenA_y-Z[i]
        S_A, R_A = precesion_TM(view, phi, theta, center_A)
        
        SRTR = np.linalg.inv(T_D @ R_D @ S_A @ R_A @ T_Z @ R_Z) @ coords
        sino3d[i,:,:] = np.squeeze(af.coords_transform(sino3d, SRTR))
    
    return sino3d
