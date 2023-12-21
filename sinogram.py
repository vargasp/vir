#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:42:09 2023

@author: pvargas21
"""


import numpy as np
import matplotlib.pyplot as plt


from scipy.optimize import curve_fit

import vir.affine_transforms as af


def cog(sino):
    return np.sum(sino*np.arange(1,sino.shape[-1]+1),axis=-1)/sino.sum(axis=-1)


def sine_wave(x, offset, amplitude, phase):
    return np.sin(x + phase) * amplitude + offset


def plot_fit(sino,angs):
    nAngs, nCols = sino.shape 
    cg = cog(sino)
    
    a = [(nCols-1)/2., (np.max(cg) - np.min(cg))/2., 0.]                               

    cf_p, pcov = curve_fit(sine_wave, angs, cg, p0=a)
    
    y = sine_wave(angs, cf_p[0], cf_p[1], cf_p[2])
    print(cf_p)
    plt.plot(angs,cg)
    plt.plot(angs,y)
    plt.show()
    
    plt.plot(angs,y-cg)
    plt.show()
    

def estimate_wobble(sino,angs):
    
    nAngs, nRows, nCols = sino.shape 
    cg = cog(sino)
    
    wave_properties = np.zeros([3,nRows])
    for row in range(nRows):
        a = [(nCols-1)/2., (np.max(cg[:,row]) - np.min(cg[:,row]))/2., 0.]                               

        cf_p, pcov = curve_fit(sine_wave, angs, cg[:,row], p0=a)
        wave_properties[:,row] = cf_p[:3]

    return wave_properties


def correct_wobble(sino, angs, phi, theta, center=None):
    """
    [nAngles,nRows,nCols]

    Parameters
    ----------
    coords : TYPE
        DESCRIPTION.
    center : TYPE
        DESCRIPTION.
    angs : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    phi : TYPE
        Angle between the principle axis of stage rotation and sample rotation

    Returns
    -------
    None.

    """
    
    nAngs, nRows, nCols = sino.shape 
    
    if center is None:
        center = (0.5, nCols/2.0 + 0.5)
        
    sino_corect = np.zeros((nAngs, nRows, nCols))
    
    
    coords = af.coords_array((nRows,nCols), ones=True)
    angle_shifts = phi*np.cos(angs+theta)
    
    for i, angle_shift in enumerate(angle_shifts):
        R = af.rotateMat(angle_shift, center=center)
        RC = (R @ coords)
        sino_corect[i,:,:] = af.coords_transform(sino[i,:,:], np.round(RC,6))
        
    return sino_corect


def forward_project_wobble(phantom, angs, phi, theta, center=None):

    
    nX, nY, nZ = phantom.shape

    if center is None:
        center = (nX/2.-.5, nY/2.-.5, 0.5)

    sino = np.zeros([angs.size,nZ,nX])
    
    coords = af.coords_array((nX,nY,nZ), ones=True)

    for i, ang in enumerate(angs):
        R = af.rotateMat((theta,phi,ang), center=center)
        RC = (R @ coords)
        sino[i,:,:] = af.coords_transform(phantom, np.round(RC,6)).sum(axis=1).T
    
    return sino    



