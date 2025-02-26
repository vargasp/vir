#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:33:22 2025

@author: pvargas21
"""

import numpy as np
import matplotlib.pyplot as plt
import vir
import vir.sino_calibration as sc
import vir.affine_transforms as af
import vt

def phantom2(f):
    nX,nY,nZ = (128,128,256)

    phantom = np.zeros([nX*f, nY*f, nZ*f], dtype=np.float32)
    phantom[56*f:72*f,56*f:72*f,8*f:248*f] = 1
    
    return phantom


def sino_par2hel(sinoP, nRows, nAngs, pitch, transD=0.0, rD=0.0):
    nViews, nRowsP, nCols = sinoP.shape
    
    dView = (Views[-1] - Views[0])/(nViews-1)
    dZ = pitch*nRows*dView/(np.pi*2)
    c =nRowsP/2 - nRows/2 +1
    Z = vir.censpace(nViews,c=c,d=dZ)

    sinoH = np.zeros([Views.size,nRows,nCols], dtype=np.float32)


    for i in range(nViews):
        coords = af.coords_array((1,nRows,nCols), ones=True)
        coords[:,:,0,:] = i
        coords[:,:,1,:] = coords[:,:,1,:] + Z[i]
        
        center=(0.0,Z[i]+nRows/2-0.5,nCols/2-0.5)
        T, R = sc.proj_orient_TM(rD, transD, center)  

        RTC = (np.linalg.inv(R @ T) @ coords)

        sinoH[i,:,:] = af.coords_transform(sinoP, np.round(RTC,6))

    return sinoH


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
sino = sc.forward_project_phantom_misalign(phantom, Views)
vt.animated_gif(sino, "sino", fps=48)

view_sinos(sino)

"""Axis of rotaion, rotated with respect to z-axis"""
angX_Z,angY_Z = (0.0,0.1)
center_Z = (nX/2.-0.5, nY/2.-0.5, .5)
sino0 = sc.forward_project_phantom_misalign(phantom, Views, \
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z)
view_sinos(sino0)

#Correction
sino0C = sc.calib_proj_orient(sino0.copy(),Views,rZ=-angY_Z, cenZ_y=center_Z[2])
view_sinos(sino0C, sino)


"""Axis of rotaion, rotated with respect to z-axis and precessing"""
angX_Z,angY_Z = (0.0,0.1)
center_Z = (nX/2.-0.5, nY/2.-0.5, .5)
angX_A,angY_A = (0.0,0.1)
center_A = (nX/2.-0.5, nY/2.-0.5, 64-.5)
sino1 = sc.forward_project_phantom_misalign(phantom, Views, \
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z,\
                                        angX_A=angX_A,angY_A=angY_A,center_A=center_A)
view_sinos(sino1)

#Correction
sino1C = sc.calib_proj_orient(sino1.copy(),Views,rZ=-angY_Z,cenZ_y=center_Z[2],\
                           phi=angY_A,cenA_y=center_A[2])
view_sinos(sino1C, sino)


"""Axis of rotaion - rotated and translated with respect to z-axis and precessing"""
trans_X = 10
angX_Z,angY_Z = (0.0,0.1)
center_Z = (nX/2.-0.5, nY/2.-0.5, .5)
angX_A,angY_A = (0.0,0.1)
center_A = (nX/2.-0.5, nY/2.-0.5, 64-.5)
sino2 = sc.forward_project_phantom_misalign(phantom, Views, trans_X=trans_X,\
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z,\
                                        angX_A=angX_A,angY_A=angY_A,center_A=center_A)
view_sinos(sino2)

#Correction
sino2C = sc.calib_proj_orient(sino2.copy(),Views,transX=-trans_X,\
                           rZ=-angY_Z,cenZ_y=center_Z[2],\
                           phi=angY_A,cenA_y=center_A[2])
view_sinos(sino2C, sino)



"""#Helical"""
sinoH = sino_par2hel(sino, 128, nAngs, 1)
view_sinos(sinoH)
vt.animated_gif(sinoH, "sinoH24", fps=48)

"""Axis of rotaion, rotated with respect to z-axis"""
sinoH0 = sino_par2hel(sino0, 128, nAngs, 1)
sino0HC = sc.calib_proj_orient(sinoH0.copy(),Views,rZ=-angY_Z,cenZ_y=center_Z[2],\
                            pitch=1)
view_sinos(sino0HC, sinoH)


"""Axis of rotaion, rotated with respect to z-axis and precessing"""
sinoH1 = sino_par2hel(sino1, 128, nAngs, 1)
sinoH1C = sc.calib_proj_orient(sinoH1.copy(),Views,rZ=-angY_Z,cenZ_y=center_Z[2],\
                            phi=angY_A,cenA_y=center_A[2],\
                            pitch=1)
view_sinos(sinoH1C, sinoH)


"""Axis of rotaion - rotated and translated with respect to z-axis and precessing"""
sinoH2 = sino_par2hel(sino2, 128, nAngs, 1)
sinoH2C = sc.calib_proj_orient(sinoH2.copy(), Views, transX=-trans_X,\
                            rZ=-angY_Z,cenZ_y=center_Z[2],\
                            phi=angY_A,cenA_y=center_A[2],\
                            pitch=1)
view_sinos(sinoH2C, sinoH)


"""Dectector offset"""
transD=5.5
sinoH_d0 = sino_par2hel(sino, 128, nAngs, 1, transD=transD)
sinoHd0C = sc.calib_proj_orient(sinoH_d0.copy(), Views, transD=-transD, pitch=1)
view_sinos(sinoHd0C, sinoH)


"""Dectector rotation"""
rD=0.075
sinoH_d1 = sino_par2hel(sino, 128, nAngs, 1, rD=rD)
sinoHd1C = sc.calib_proj_orient(sinoH_d1.copy(), Views, rD=-rD, pitch=1)
view_sinos(sinoHd1C, sinoH)


"""Dectector offset, Dectector rotation"""    
sinoH_d2 = sino_par2hel(sino, 128, nAngs, 1, transD=transD,rD=rD)
sinoHd2C = sc.calib_proj_orient(sinoH_d2.copy(), Views, transD=-transD, rD=-rD,\
                             pitch=1)
view_sinos(sinoHd2C, sinoH)


"""Axis of rotaion - rotated and translated with respect to z-axis and precessing"""
"""Dectector offset, Dectector rotation"""    
sinoH2_d2 = sino_par2hel(sino2, 128, nAngs, 1, transD=transD,rD=rD)
sinoH2_d2C = sc.calib_proj_orient(sinoH2_d2.copy(), Views, transD=-transD, rD=-rD,\
                               transX=-trans_X,\
                               rZ=-angY_Z,cenZ_y=center_Z[2],\
                               phi=angY_A,cenA_y=center_A[2],\
                               pitch=1)
view_sinos(sinoH2_d2C, sinoH)

