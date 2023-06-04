#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:33:57 2023

@author: pvargas21
"""

import numpy as np

import vir

def gaussian1d(mu=0.0, sigma=5.0, A=None, nX=128):
    """
    Generates a 1-d gaussian function.

    Parameters
    ----------
    mu : int, optional
        The mean value of the gaussian function. The default is 0.0.
    sigma : int, optional
        The standard deviation of the gaussain function. The default is 5.0.
    A : int, optional
        The amplitude of the function. If None is provided the amplituded will
        provide a normalized function. The default is None.
    nX : int, optional
        The number of pixels for the function. The default is 128.

    Returns
    -------
    (nX) numpy ndarray
        The gaussain function 
    """
    
    #Generates the X axis
    x = vir.censpace(nX)

    #If the amplitude is not provided calcualtes a normalized amplitude
    if A is None:
        A = 1.0 / (sigma*np.sqrt(2*np.pi))
            
    #Returns the kernel
    return A*np.exp(-(x - mu)**2/(2*sigma**2))


def gaussian2d(mus=(0.0,0.0), sigmas=(5.0,5.0), theta=0, A=None, \
               nX=128, nY=128):
    
    x_mu,y_mu = mus
    x_sigma, y_sigma = sigmas
    
    x = vir.censpace(nX)
    y = vir.censpace(nY)
    x, y = np.meshgrid(x,y)
    
    a = np.cos(theta)**2 / (2*x_sigma**2) + np.sin(theta)**2 / (2*y_sigma**2)
    b = -np.sin(2*theta) / (4*x_sigma**2) + np.sin(2*theta) / (4*y_sigma**2)
    c = np.sin(theta)**2 / (2*x_sigma**2) + np.cos(theta)**2 / (2*y_sigma**2)
    
    if A is None:
        A = 1.0 / (x_sigma*y_sigma * (2*np.pi))
    
    return A*np.exp(-a*(x - x_mu)**2 - 2*b*(x - x_mu)*(y - y_mu) - c*(y - y_mu)**2)
    


def gaussian3d(mus=(0.0,0.0,0.0), sigmas=(5.0,5.0,5.0), A=None, \
               nX=128, nY=128, nZ=128):
    
    x_mu, y_mu, z_mu = mus
    x_sigma, y_sigma, z_sigma = sigmas
    
    x = vir.censpace(nX)
    y = vir.censpace(nY)
    z = vir.censpace(nZ)
    x, y, z = np.meshgrid(x,y,z)
    
    if A is None:
        A = 1.0 / (x_sigma*y_sigma*z_sigma * (2*np.pi)**(3/2))
            
    return A*np.exp(-(x - x_mu)**2/(2*x_sigma**2) \
                    -(y - y_mu)**2/(2*y_sigma**2) \
                    -(z - z_mu)**2/(2*z_sigma**2))
        

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
