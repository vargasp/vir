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
    return np.sum(sino*np.arange(1,sino.shape[-1]+1),axis=-1)/sino.sum(axis=-1) - .5


def sine_wave(x, offset, amplitude, phase):
    return np.cos(x + phase) * amplitude + offset


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
    


def _org_wave_properites(wp):
    
    idx = wp[2,:] < 0
    wp[2,idx] +=  2*np.pi

    idx = wp[2,:] > 2*np.pi
    wp[2,idx] %= (2*np.pi)
    
    return wp


def org_wave_properites(wp):
    
    wp = _org_wave_properites(wp)
        
    #Deterines how many modes are in the the theta array
    bins = np.linspace(0,np.pi*2,8,endpoint=True)
    hist = np.histogram(wp[2,:],bins=bins)
    nModes = np.sum(hist[0] > 0)
    
    
    if nModes == 2:
        
        idxs = np.argsort(hist[0])[-2:]

        m0_sub = (wp[2,:] > hist[1][idxs[0]]) * (wp[2,:] < hist[1][idxs[0]+1])
        m0_mean = np.mean(wp[2,m0_sub])

        m1_sub = (wp[2,:] > hist[1][idxs[1]]) * (wp[2,:] < hist[1][idxs[1]+1])
        m1_mean = np.mean(wp[2,m1_sub])


        #Aligns thetas accros slices
        if np.isclose(m0_mean-m1_mean, np.pi*2, atol=0.01):
            wp[2,m0_sub] -= np.pi*2
    
        if np.isclose(m1_mean-m0_mean, np.pi*2, atol=0.01):
            wp[2,m1_sub] -= np.pi*2
    
        if np.isclose(m1_mean-m0_mean, np.pi, atol=0.01):
            wp[2,m0_sub] += np.pi
            wp[1,m0_sub] *= -1.0
    
        if np.isclose(m0_mean-m1_mean, np.pi, atol=0.01):
            wp[2,m0_sub] -= np.pi
            wp[1,m0_sub] *= -1.0
    
    slope = np.polyfit(np.arange(wp.shape[1]), wp[1,:], 1)[0]
    if slope < 0:
        wp[2,:] += np.pi
        wp[1,:] *= -1.0
        wp = _org_wave_properites(wp)
    
    
    return wp


def estimate_wobble(sino,angs):
    
    nAngs, nRows, nCols = sino.shape 
    cg = cog(sino)
    
    wave_properties = np.zeros([3,nRows])
    #bounds = ((0,0,0),(nCols,4*nCols,2*np.pi))
    
    for row in range(nRows):
        a = [(nCols-1)/2., (np.max(cg[:,row]) - np.min(cg[:,row]))/2., angs[np.argmax(angs)]]                               

        #cf_p, pcov = curve_fit(sine_wave, angs, cg[:,row], p0=a, bounds=bounds)
        cf_p, pcov = curve_fit(sine_wave, angs, cg[:,row], p0=a)
                
        wave_properties[:,row] = cf_p[:3]


    wave_properties = org_wave_properites(wave_properties)
        

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
    angle_shifts = -phi*np.cos(angs+theta)
    
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



