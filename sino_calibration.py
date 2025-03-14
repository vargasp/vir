#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:33:06 2025

@author: pvargas21
"""

import numpy as np

import vir
import vir.affine_transforms as af
from scipy.ndimage import sobel
import vt


def forward_project_phantom_misalign(phantom, Views, \
                                     transX=0.0,transY=0.0,transZ=0.0,
                                     angZx=0.0, angZy=0.0, centerZz=0.0,\
                                     angAx=0.0, angAy=0.0, centerAz=0.0,
                                     ret_phantom=False):
    """
    Creates a paralell beam sinogram from a 3d phantom with a misalignments.
    The phantom is projected in the XZ plane while rotating.

    Parameters
    ----------
    phantom : (nX, nY, nZ) np.array
        A 3d phantom.
    Views : (nViews) np.array
        The array of angles of rotation. Rotation is counter clockwize in the
        XY plane with the z-axis toward the viewer 
    transX : float
        Translates the axis of rotation in the X dimension.The default is 0.0
        or no translation.
    transY : float
        Translates the axis of rotation in the Y dimension.The default is 0.0
        or no translation. (Has no affect in paralell beam geometry)
    angZx : float
        Rotates the axis of rotation in YZ plane relative to the z-axis.The default is 0.0
        or no rotation which aligns with z-axis. Postive values rotate counter clockwise in the YZ plane
    angZy : float
        Rotates the axis of rotation in XZ plane relative to the z-axis.The default is 0.0
        or no rotation which aligns with z-axis. Postive values rotate counter clockwise in the XZ plane
    center : (3) array_like, optional
        The center of rotation location. The default is "None" which
        corresponds to the center of the image.

    Returns
    -------
    sino : (nAngs, nY, nZ) np.array
        The forward projected sinogram with a precession artifact.
    """

    nX, nY, nZ = phantom.shape

    centerA = np.array(phantom.shape)/2.0 - 0.5
    centerA[2] += centerAz
    
    centerZ = np.array(phantom.shape)/2.0 - 0.5
    centerZ[2] += centerZz

    
    sino = np.zeros([Views.size,nZ,nX], dtype=np.float32)
    coords = af.coords_array((nX,nY,nZ), ones=True)

    #Misalign phantom by translating axis of rotation with respect to z-axis
    T = af.transMat((-transX,-transY,-transZ))

    #Misalign phantom by rotating axis of rotation with respect to z-axis
    R_Z = af.rotateMat((-angZx,angZy,0), center=centerZ, seq='XYZ')

    #Loop through views
    for i, view in enumerate(Views):

        #Rotate for tomographic projection and add precession misalignment  
        R_A = af.rotateMat((-angAx,angAy,-view), center=centerA, seq='XYZ')

        #Apply transforms and project view
        RRTC = np.round(R_A @ R_Z @ T @ coords,6)
        
        if ret_phantom: return af.coords_transform(phantom, RRTC)
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


def calib_proj_orient(sino3d, Views, transX=0.0, rZ=0.0, centerZz=0.0,\
                      phi=0.0, theta=0.0, centerAz=0.0, transD=0.0, rD=0.0,\
                      pitch=0):
    nViews, nRows, nCols = sino3d.shape
    

    centerD = np.array((0.0, nRows/2.-0.5, nCols/2.-0.5))
    centerA = centerD.copy()
    centerZ = centerD.copy()
    
    T_D, R_D = proj_orient_TM(rD, transD, centerD)  
    
    
    dView = (Views[-1] - Views[0])/(nViews-1)
    dZ = pitch*nRows*dView/(2.0*np.pi)    
    Z = vir.censpace(nViews,d=dZ)

    
    coords = af.coords_array((1,nRows,nCols), ones=True)
    #coords[:,:,0,:] = np.interp(ang,Angs,np.arange(nViews))
    

    for i in range(nViews):
        coords[:,:,0,:] = i
        view = Views[i]

        #Center of of rotation transforms
        centerZ[1] = nRows/2.-0.5 + centerZz - Z[i]
        T_Z, R_Z = proj_orient_TM(rZ, -transX, centerZ)

        #Precession Transforms
        centerA[1] = nRows/2.-0.5 + centerAz - Z[i]
        S_A, R_A = precesion_TM(view, phi, theta, centerA)
        
        SRTR = np.linalg.inv(T_D @ R_D @ S_A @ R_A @ T_Z @ R_Z) @ coords
        sino3d[i,:,:] = np.squeeze(af.coords_transform(sino3d, SRTR))
    
    return sino3d


def hsino_calibrate(sino,Views,pitch,transX,rZ,cenZ_y,phi,theta,cenA_y,rD,cL=38,\
          Z=0,par=False):
    nViews, nRows, nCols = sino.shape
    nAngs = np.interp(2*np.pi,Views,np.arange(nViews))
        
    center_Z = np.array((0.0, nRows/2.-0.5, nCols/2.-0.5))
    
    p_sino = np.zeros([nViews,nCols])
    h_sino = np.zeros([nViews,nCols,2])
    
    coords = af.coords_array((1,3,nCols), ones=True)
    coords[:,0,1,:] = cL
    coords[:,1,1,:] = cL + pitch*nRows/2.0

    
    center_det = np.array([1.0,nRows,nCols])/2.0 - 0.5
    x=0
    T_det, R_D = proj_orient_TM(rD, x, center_det)
    
    dView = (Views[-1] - Views[0])/(nViews-1)
    dZ = pitch*nRows*dView/(2.0*np.pi)    
    Z = vir.censpace(nViews,d=dZ)
    
    
    dZ1 = pitch*nRows*Views/(2.0*np.pi)       
    dZ2 = pitch*nRows*(Views-np.pi)/(2.0*np.pi) 

    for i, view in enumerate(Views):
        coords[:,0,0,:] = i
        coords[:,1,0,:] = i - nAngs/2.0
        #coords[:,2,0,:] = i
        #coords[:,2,1,:] = Z - dZ1[i]

        #Center of of rotation transforms        
        center_Z[1] = nRows/2.-0.5 + cenZ_y - Z[i]
        T_Z, R_Z = proj_orient_TM(rZ, -transX, center_Z)
    
        center_A = np.array((0.0, cenA_y-dZ1[i], nCols/2.-0.5))
        S_A, R_A = precesion_TM(view, phi, theta, center_A)
    
        SRTR1 = np.linalg.inv(R_D @ S_A @ R_A @T_Z @ R_Z) 
        h_sino[i,:,0] = np.squeeze(af.coords_transform(sino, SRTR1 @ coords[:,[0],:,:]))
        p_sino[i,:] = np.squeeze(af.coords_transform(sino, SRTR1@ coords[:,[2],:,:]))

        #Center of of rotation transforms
        #center_Z[1] = nRows/2.-0.5 + centerZz - Z[i]
        #T_Z, R_Z = proj_orient_TM(-rZ, -transX, center_Z)
    
        #center_A = np.array((0.0, cenA_y-dZ2[i], nCols/2.-0.5))
        #S_A, R_A = precesion_TM(view-np.pi, phi, theta, center_A)
        
        #SRTR2 = np.linalg.inv(R_D @ S_A @ R_A @T_Z @ R_Z) @ coords[:,[1],:,:]
        #h_sino[i,:,1] =  np.squeeze(af.coords_transform(sino, SRTR2))[::-1]


    if par is True:
        return h_sino[int(nAngs/2):-int(nAngs/2),:,:],p_sino
    else:
        #return h_sino[int(nAngs/2):-int(nAngs/2),:,:]
        return h_sino[:-int(nAngs/2),:,:]


"""
Old functions
def hsino_calibrate(sino,nAngs,pitch,transX,rZ,cenZ_y,phi,theta,cenA_y,rD,cL=38,\
          Z=0,par=False):
    nViews, nRows, nCols = sino.shape
    Views = np.linspace(0,nViews/nAngs*np.pi*2,nViews,endpoint=False, dtype=np.float32)
        
    p_sino = np.zeros([nViews,nCols])
    h_sino = np.zeros([nViews+int(nAngs/2),nCols,2])
    
    coords = af.coords_array((1,3,nCols), ones=True)
    coords[:,0,1,:] = cL
    coords[:,1,1,:] = cL + pitch*nRows/2.0

    
    center_det = np.array([1.0,nRows,nCols])/2.0 - 0.5
    x=0
    T_det, R_D = proj_orient_TM(rD, x, center_det)
    
    dZ1 = pitch*nRows*Views/(2*np.pi)       
    dZ2 = pitch*nRows*(Views-np.pi)/(2*np.pi) 

    for i, view in enumerate(Views):
        coords[:,0,0,:] = i
        coords[:,1,0,:] = i - nAngs/2.0
        coords[:,2,0,:] = i
        coords[:,2,1,:] = Z - dZ1[i]

        #Center of of rotation transforms        
        center_Z = np.array((0.0, cenZ_y-dZ1[i], nCols/2.-0.5))
        T_Z, R_Z = proj_orient_TM(-rZ, -transX, center_Z)
    
        center_A = np.array((0.0, cenA_y-dZ1[i], nCols/2.-0.5))
        S_A, R_A = precesion_TM(view, phi, theta, center_A)
    
        SRTR1 = np.linalg.inv(R_D @ S_A @ R_A @T_Z @ R_Z) 
        h_sino[i,:,0] = np.squeeze(af.coords_transform(sino, SRTR1 @ coords[:,[0],:,:]))
        p_sino[i,:] = np.squeeze(af.coords_transform(sino, SRTR1@ coords[:,[2],:,:]))

        #Center of of rotation transforms        
        center_Z = np.array((0.0, cenZ_y-dZ2[i], nCols/2.-0.5))
        T_Z, R_Z = proj_orient_TM(-rZ, -transX, center_Z)
    
        center_A = np.array((0.0, cenA_y-dZ2[i], nCols/2.-0.5))
        S_A, R_A = precesion_TM(view-np.pi, phi, theta, center_A)
        
        SRTR2 = np.linalg.inv(R_D @ S_A @ R_A @T_Z @ R_Z) @ coords[:,[1],:,:]
        h_sino[i,:,1] =  np.squeeze(af.coords_transform(sino, SRTR2))[::-1]


    if par is True:
        return h_sino[int(nAngs/2):-int(nAngs/2),:,:],p_sino
    else:
        #return h_sino[int(nAngs/2):-int(nAngs/2),:,:]
        return h_sino[:-int(nAngs/2),:,:]
"""


def rmse(sino):
    return np.sqrt(np.sum((sino[:,:,0] - sino[:,:,1])**2)/sino[:,:,0].size)

def plot_calib_args(*args):
    sino = args[0]
    
    args = list(args)
    for i in range(1,10):
        if type(args[i]) is not np.ndarray: args[i] = [args[i]]
    
    nAngsV = np.array(args[1])
    pitchV = np.array(args[2])
    transXV = np.array(args[3])
    rZV = np.array(args[4])
    cenZ_yV = np.array(args[5])
    phiV = np.array(args[6])
    thetaV = np.array(args[7])
    cenA_yV = np.array(args[8])
    rDV = np.array(args[9])
    
    rmat = np.zeros([nAngsV.size,pitchV.size,transXV.size,\
                     rZV.size,cenZ_yV.size,\
                     phiV.size,thetaV.size,cenA_yV.size,rDV.size,2])
    
    for i, nAngs in enumerate(nAngsV):
        for j, pitch in enumerate(pitchV):
            for k, transX in enumerate(transXV):
                for l, rZ in enumerate(rZV):
                    for m, cenZ_y in enumerate(cenZ_yV):
                        for n, phi in enumerate(phiV):
                            for o, theta in enumerate(thetaV):
                                for p, cenA_y in enumerate(cenA_yV):
                                    for q, rD in enumerate(rDV):
    
                                        test2 = hsino_calibrate(sino,nAngs,pitch,transX,rZ,cenZ_y,phi,theta,cenA_y,rD)
                                        rmat[i,j,k,l,m,n,o,p,q,0] = rmse(sobel(test2,axis=0))
                                        rmat[i,j,k,l,m,n,o,p,q,1] = rmse(sobel(test2,axis=1))


    rmat = rmat.squeeze()
    i0,i1 = np.argmin(rmat, axis=0)
    mx0,mx1 = np.max(rmat, axis=0)
    mn0,mn1 = np.min(rmat, axis=0)

    if nAngsV.size >1:
        print("nAngs 0:",nAngsV[i0], "Idx:",i0, "RMSE:",mn0, "Range:",mx0-mn0)
        #print("pitch 1:",pitchV[i1], "Idx:",i1, "RMSE:",mn1, "Range:",mx1-mn1)
        vt.CreatePlot(rmat[:,0],xs=nAngsV,xtitle='nAngs',ytitle='RMSE')
    if pitchV.size >1:
        print("pitch 0:",pitchV[i0], "Idx:",i0, "RMSE:",mn0, "Range:",mx0-mn0)
        #print("pitch 1:",pitchV[i1], "Idx:",i1, "RMSE:",mn1, "Range:",mx1-mn1)
        vt.CreatePlot(rmat[:,0],xs=pitchV,xtitle='pitch',ytitle='RMSE')
    if transXV.size >1:
        print("transX 0:",transXV[i0], "Idx:",i0, "RMSE:",mn0, "Range:",mx0-mn0)
        #print("transX 1:",transXV[i1], "Idx:",i1, "RMSE:",mn1, "Range:",mx1-mn1)
        vt.CreatePlot(rmat[:,0],xs=transXV,xtitle='transX',ytitle='RMSE')
    if rZV.size >1:
        print("rZ 0:",rZV[i0], "Idx:",i0, "RMSE:",mn0, "Range:",mx0-mn0)
        #print("rZ 1:",rZV[i1], "Idx:",i1, "RMSE:",mn1, "Range:",mx1-mn1)
        vt.CreatePlot(rmat[:,0],xs=rZV,xtitle='rZ',ytitle='RMSE')
    if cenZ_yV.size >1:
        print("cenZ_y 0:",cenZ_yV[i0], "Idx:",i0, "RMSE:",mn0, "Range:",mx0-mn0)
        #print("cenZ_y 1:",cenZ_yV[i1], "Idx:",i1, "RMSE:",mn1, "Range:",mx1-mn1)
        vt.CreatePlot(rmat[:,0],xs=cenZ_yV,xtitle='cenZ_y',ytitle='RMSE')
    if phiV.size >1:
        print("phi 0:",phiV[i0], "Idx:",i0, "RMSE:",mn0, "Range:",mx0-mn0)
        #print("phi 1:",phiV[i1], "Idx:",i1, "RMSE:",mn1, "Range:",mx1-mn1)
        vt.CreatePlot(rmat[:,0],xs=phiV,xtitle='phi',ytitle='RMSE')
    if thetaV.size >1:
        print("theta 0:",thetaV[i0], "Idx:",i0, "RMSE:",mn0, "Range:",mx0-mn0)
        #print("theta 1:",thetaV[i1], "Idx:",i1, "RMSE:",mn1, "Range:",mx1-mn1)
        vt.CreatePlot(rmat[:,0],xs=thetaV,xtitle='theta',ytitle='RMSE')
    if cenA_yV.size >1:
        print("cenA_y 0:",cenA_yV[i0], "Idx:",i0, "RMSE:",mn0, "Range:",mx0-mn0)
        #print("cenA_y 1:",cenA_yV[i1], "Idx:",i1, "RMSE:",mn1, "Range:",mx1-mn1)
        vt.CreatePlot(rmat[:,0],xs=cenA_yV,xtitle='cenA_y',ytitle='RMSE')
    if rDV.size >1:
        print("rDV0:",rDV[i0], "Idx:",i0, "RMSE:",mn0, "Range:",mx0-mn0)
        #print("cenA_y 1:",cenA_yV[i1], "Idx:",i1, "RMSE:",mn1, "Range:",mx1-mn1)
        vt.CreatePlot(rmat[:,0],xs=rDV,xtitle='rDV',ytitle='RMSE')
    return rmat


#from scipy.optimize import minimize
#params = (nAngs,pitch,transX,rZ,cenZ_y,phi,theta,cenA_y,rD)
#results = minimize(min_claib, params, args=(sino),method='Nelder-Mead')

def min_params(params, sino):
    nAngs,pitch,transX,rZ,cenZ_y,phi,theta,cenA_y,rD = params
    

    hsino = hsino_calibrate(sino,nAngs,pitch,transX,rZ,cenZ_y,phi,theta,cenA_y,rD)

    return rmse(hsino)
    





