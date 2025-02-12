#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:33:22 2025

@author: pvargas21
"""

import numpy as np
import matplotlib.pyplot as plt
import vir.affine_transforms as af


def phantom2(f):
    nX,nY,nZ = (128,128,256)

    phantom = np.zeros([nX*f, nY*f, nZ*f], dtype=np.float32)
    phantom[56*f:72*f,56*f:72*f,8*f:248*f] = 1
    
    return phantom


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

    T = af.transMat((trans_X,trans_Y,trans_Z))
    R_Z = af.rotateMat((angX_Z,angY_Z,0), center=center_Z, seq='XYZ')


    for i, view in enumerate(Views):
        print(i)
        R_A = af.rotateMat((angX_A,angY_A,view), center=center_A, seq='XYZ')
        RRTC = (R_A @ R_Z @ T @ coords)

        sino[i,:,:] = af.coords_transform(phantom, np.round(RRTC,6)).sum(axis=1).T
    
    return sino    


def sino_par2hel(sinoP, nRows, nAngs, pitch):
    nViews, nRowsP, nCols = phantom.shape
    
    coords = af.coords_array((1,nRows,nCols), ones=True)
    sinoH = np.zeros([Views.size,nRows,nCols], dtype=np.float32)

    R = af.rotateMat((0,0,0), center=(nAngs/2-0.5,nRows/2-0.5,nCols/2-0.5))

    for i in range(nViews):
        T = af.transMat((0,-i,0))
        coords[:,:,0,:] = i
        RTC = (np.linalg.inv(R @ T) @ coords)
    
        sinoH[i,:,:] = af.coords_transform(sinoP, np.round(RTC,6))

    return sinoH




"""#Helical"""
phantom = phantom2(1)
nX, nY, nZ = phantom.shape

nAngs = 256
nRots = 1
Views = np.linspace(0, nRots*2*np.pi,nRots*nAngs,endpoint=False, dtype=np.float32)

angX_Z = 0.0
angY_Z = 0.1
center_Z = (nX/2.-0.5, nY/2.-0.5, .5)
sino0 = forward_project_phantom_misalign(phantom, Views, \
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z)

for view in np.arange(16)*16:
    plt.imshow(sino0[view,:,:], origin='lower')
    plt.show()


angX_Z = 0.0
angY_Z = 0.1
center_Z = (nX/2.-0.5, nY/2.-0.5, .5)
angX_A = 0.0
angY_A = 0.1
center_A = (nX/2.-0.5, nY/2.-0.5, 64-.5)
sino1 = forward_project_phantom_misalign(phantom, Views, \
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z,\
                                        angX_A=angX_A,angY_A=angY_A,center_A=center_A)
    
for view in np.arange(16)*16:
    plt.imshow(sino1[view,:,:], origin='lower')
    plt.show() 


trans_X = 10
angX_Z = 0.0
angY_Z = 0.1
center_Z = (nX/2.-0.5, nY/2.-0.5, .5)
angX_A = 0.0
angY_A = 0.1
center_A = (nX/2.-0.5, nY/2.-0.5, 64-.5)
sino2 = forward_project_phantom_misalign(phantom, Views, trans_X=trans_X,\
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z,\
                                        angX_A=angX_A,angY_A=angY_A,center_A=center_A)
    
for view in np.arange(16)*16:
    plt.imshow(sino2[view,:,:], origin='lower')
    plt.show() 



sinoH0 = sino_par2hel(sino2, 128, nAngs, 1)





