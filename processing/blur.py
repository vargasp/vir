#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:23:21 2020

@author: vargasp
"""
import numpy as np


def LorentzianPSF(gamma,dPixel=1.0,dims=1,epsilon=1e-3):
    """
    Returns the resized array of the specified dimensions.
    
    Parameters
    ----------
    gamma : float
        The FWHM of the Lorentzian function
    dPixe; : float or (2) array_like
        The size of the pixels in the kernel (sampling interval) in the X and
        Y dimensions. Default is 1.0
    dims : int
        The number of dimensions 
    epsilon; : float 
        The fractional cutoff of the function tail. Decreasing this value
        increases the size of the kernel. Defalt is 1e-3

    Returns
    -------
    psf :  numpy ndarray 
        The computed Lorentzian function in an odd kernel
    """

    #Calculates the size of kernel based on epsilon
    span = np.sqrt( (0.5*gamma)**2*(1.0/epsilon - 1.0))    
    
    if dims == 1:
        dX = dPixel
        nX = int(np.ceil(span/dPixel) // 2 * 2 + 1)

        X = np.linspace(-nX+1,nX-1,nX)*dX/2.0

        psf = 1.0/np.pi * 0.5*gamma / (X**2 + (0.5*gamma)**2)
        
    elif dims == 2:
        dPixel = np.array(dPixel,dtype=float)
        if dPixel.size == 1:
            dPixel = np.repeat(dPixel,2)
    
        dX,dY, = dPixel
        
        nX = int(np.ceil(span/dX) // 2 * 2 + 1)
        nY = int(np.ceil(span/dY) // 2 * 2 + 1)

        X = np.linspace(-nX+1,nX-1,nX)*dX/2.0
        Y = np.linspace(-nY+1,nY-1,nY)*dY/2.0

        XX, YY = np.meshgrid(X, Y)
    
        psf = 1.0/np.pi * 0.5*gamma / (XX**2 + YY**2 + (0.5*gamma)**2)

    #Retruns normalized point spread function
    return psf/psf.sum()
