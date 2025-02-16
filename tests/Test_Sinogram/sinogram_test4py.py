#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:33:22 2025

@author: pvargas21
"""

import numpy as np
import matplotlib.pyplot as plt
import vir.affine_transforms as af
import vir

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

    #Misalign phantom by translating axis of rotation with respect to z-axis
    T = af.transMat((trans_X,trans_Y,trans_Z))

    #Misalign phantom by rotating axis of rotation with respect to z-axis
    R_Z = af.rotateMat((angX_Z,angY_Z,0), center=center_Z, seq='XYZ')

    #Loop through views
    for i, view in enumerate(Views):

        #Rotate for tomographic projection and add precession misalignment  
        R_A = af.rotateMat((angX_A,angY_A,view), center=center_A, seq='XYZ')

        #Apply transforms and project view
        RRTC = (R_A @ R_Z @ T @ coords)
        sino[i,:,:] = af.coords_transform(phantom, np.round(RRTC,6)).sum(axis=1).T
    
    return sino    





def calib_proj_orient_TM(r, x, center):
    T = af.transMat((0,0,x))
    R = af.rotateMat((r,0,0), center=center)
    return T, R


def calib_precesion_TM(ang, phi, theta, center):
    r = phi*np.cos(ang + theta)
    z = np.sin(np.pi/2 - phi)
    h_xy = np.cos(np.pi/2 - phi) 
    
    x = np.cos(ang + theta)*h_xy
    s = np.sqrt(x**2 + z**2)

    S = af.scaleMat((1,1/s,1))
    R = af.rotateMat((r,0,0), center=center)
    return S, R


def calib_proj_orient(sino3d, Angs, ang, x, rZ, cenZ_y, phi, theta, cenA_y, pitch=0):
    nViews, nRows, nCols = sino3d.shape
    
    coords = af.coords_array((1,nRows,nCols), ones=True)
    #coords[:,:,0,:] = np.interp(ang,Angs,np.arange(nViews))
    

    for i in range(nViews):
        coords[:,:,0,:] = i
        ang = Angs[i]

        #Center of of rotation transforms
        center_Z = np.array([0.0,cenZ_y,nCols/2.0 - 0.5])
        T_Z, R_Z = calib_proj_orient_TM(-rZ, -x, center_Z)

        
        #Precession Transforms
        center_A = np.array([0.0,cenA_y,nCols/2.0 - 0.5])
        S_A, R_A = calib_precesion_TM(ang, phi, theta, center_A)
        
        SRTR = np.linalg.inv(S_A @ R_A @ T_Z @ R_Z) @ coords
        sino3d[i,:,:] = np.squeeze(af.coords_transform(sino3d, SRTR))
    
    return sino3d


def view_sinos(s1, s2=None):
    nViews, nRows, nCols = s1.shape
    
    for view in np.arange(16)*int(nViews/16):
        if s2 is None: 
            plt.imshow(s1[view,:,:], origin='lower')
        else:
            plt.imshow(s1[view,:,:]-s2[view,:,:], origin='lower')
            
        plt.show()
    


"""Paralell"""
phantom = phantom2(1)
nX, nY, nZ = phantom.shape

nAngs = 256
nRots = 3
nViews = nRots*nAngs
Views = np.linspace(0, nRots*2*np.pi,nViews,endpoint=False, dtype=np.float32)
sino = forward_project_phantom_misalign(phantom, Views)


"""Axis of rotaion, rotated with respect to z-axis"""
angX_Z,angY_Z = (0.0,0.1)
center_Z = (nX/2.-0.5, nY/2.-0.5, .5)
sino0 = forward_project_phantom_misalign(phantom, Views, \
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z)
view_sinos(sino0)

#Correction
sino0C = calib_proj_orient(sino0.copy(),Views,1,0,-angY_Z,center_Z[2],0,0,0)
view_sinos(sino0C, sino)


"""Axis of rotaion, rotated with respect to z-axis and precessing"""
angX_Z,angY_Z = (0.0,0.1)
center_Z = (nX/2.-0.5, nY/2.-0.5, .5)
angX_A,angY_A = (0.0,0.1)
center_A = (nX/2.-0.5, nY/2.-0.5, 64-.5)
sino1 = forward_project_phantom_misalign(phantom, Views, \
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z,\
                                        angX_A=angX_A,angY_A=angY_A,center_A=center_A)
view_sinos(sino1)

#Correction
sino1C = calib_proj_orient(sino1.copy(),Views,1,0,-angY_Z,center_Z[2],angY_A,0,center_A[2])
view_sinos(sino1C, sino)



"""Axis of rotaion - rotated and translated with respect to z-axis and precessing"""
trans_X = 10
angX_Z,angY_Z = (0.0,0.1)
center_Z = (nX/2.-0.5, nY/2.-0.5, .5)
angX_A,angY_A = (0.0,0.1)
center_A = (nX/2.-0.5, nY/2.-0.5, 64-.5)
sino2 = forward_project_phantom_misalign(phantom, Views, trans_X=trans_X,\
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z,\
                                        angX_A=angX_A,angY_A=angY_A,center_A=center_A)
view_sinos(sino2)

#Correction
sino2C = calib_proj_orient(sino2.copy(),Views,1,-trans_X,-angY_Z,center_Z[2],angY_A,0,center_A[2])
view_sinos(sino2C, sino)

"""#Helical"""
"""Axis of rotaion, rotated with respect to z-axis"""



def sino_par2hel(sinoP, nRows, nAngs, pitch):
    nViews, nRowsP, nCols = sinoP.shape
    
    dView = (Views[-1] - Views[0])/(nViews-1)
    dZ = pitch*nRows*dView/(np.pi*2)
    Z = vir.censpace(nViews,c=nRowsP/2 - nRows/2 +1,d=dZ)
    
    coords = af.coords_array((1,nRows,nCols), ones=True)
    sinoH = np.zeros([Views.size,nRows,nCols], dtype=np.float32)

    R = af.rotateMat((0,0,0), center=(nAngs/2-0.5,nRows/2-0.5,nCols/2-0.5))

    for i in range(nViews):
        T = af.transMat((-i,-Z[i],0))
        print(Z[i])
        RTC = (np.linalg.inv(R @ T) @ coords)
    
        sinoH[i,:,:] = af.coords_transform(sinoP, np.round(RTC,6))

    return sinoH

sinoH = sino_par2hel(sino, 128, nAngs, 1)
#view_sinos(sinoH)

plt.imshow(sinoH[256,:,:]- np.flipud(sinoH[512,:,:]), origin='lower')
plt.show()


sino2C = calib_proj_orient(sinoH0.copy(),Views,1,-trans_X,-angY_Z,.5,angY_A,0,64-.5)
view_sinos(sino2C)




sinoH0 = sino_par2hel(sino2, 128, nAngs, 1)


for view in np.arange(16)*16:
    plt.imshow(sinoH0[view,:,:], origin='lower')
    plt.show() 

