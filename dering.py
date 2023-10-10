#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:11:51 2022

@author: vargasp
"""

import numpy as np
from skimage.transform import warp_polar
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import median_filter


# Auxiliary function to map polar data to a cartesian plane
def polar_to_cart(polar_data, nX, nY, theta_step=1, range_step=1,order=3):

    # "x" and "y" are numpy arrays with the desired cartesian coordinates
    # we make a meshgrid with them
    X, Y = np.meshgrid(np.arange(nX) - nX/2, np.arange(nY) - nY/2)

    # Now that we have the X and Y coordinates of each point in the output plane
    # we can calculate their corresponding theta and range
    Tc = np.degrees(np.arctan2(Y, X)).ravel()
    Rc = (np.sqrt(X**2 + Y**2)).ravel()

    # Negative angles are corrected
    Tc[Tc < 0] = 360 + Tc[Tc < 0]

    # Using the known theta and range steps, the coordinates are mapped to
    # those of the data grid
    Tc = Tc / theta_step
    Rc = Rc / range_step

    # An array of polar coordinates is created stacking the previous arrays
    coords = np.vstack((Tc, Rc))

    # To avoid holes in the 360º - 0º boundary, the last column of the data
    # copied in the begining
    polar_data = np.vstack((polar_data, polar_data[-1,:]))

    # The data is mapped to the new coordinates
    # Values outside range are substituted with nans
    cart_data = map_coordinates(polar_data, coords, order=order, mode='constant', cval=0.0)

    # The data is reshaped and returned
    return(cart_data.reshape(nY, nX))


def polar_filter(img, kernel_max, axis, steps=3):

    #Generates the kernel sizes and indices for the concentric anulii
    kernels = np.rint(kernel_max*np.arange(1,steps+1)/steps).astype(int)
    idxs = np.rint(img.shape[axis]*np.arange(0,steps+1)/steps).astype(int)

    #Generates kernel footprints and modes
    if axis == 0: #Filters in the azimuthal dimension
        kernels = list(zip(kernels,np.ones(steps, dtype=int)))
        mode = 'wrap'
    elif axis == 1: #Filters in the radial dimension
        kernels = list(zip(np.ones(steps, dtype=int),kernels))
        mode = 'reflect'
    else:
        raise ValueError("Only values 0 or 1 allowed for axis")

    #Filters the image at each annulus
    lf_img = np.zeros(img.shape)    
    for i, kernel in enumerate(kernels):            
        lf_img[:, idxs[i]:idxs[i+1]] = median_filter(img, kernel, mode=mode)[:, idxs[i]:idxs[i+1]]

    return lf_img


def dering(img,thresh_min=None,thresh_max=None,thresh_art_max=30,\
           thresh_art_min=-30, azimuthal_kernel=11,radial_kernel=21, return_art=False):

    img = img.astype(float)
    
    nX, nY = img.shape
    
    #Lower threshold for image segmentation
    if thresh_min is None: thresh_min = img.min()

    #Upper threshold for image segmentation
    if thresh_max is None: thresh_max = img.max()

    #Thresholded image
    tImg = img.clip(thresh_min,thresh_max)

    #Image transformed into polar coordinates [nThetas,nRadii]
    nRadii = int(np.ceil(np.sqrt((nX/2)**2 + (nY/2)**2)))
    nThetas = 1440
    pImg = warp_polar(tImg, output_shape=[nThetas,nRadii]) 

    #Radial median filtering in polar coordinates
    fImg = polar_filter(pImg, radial_kernel, 1)
    
    #Subtract median image to calculate artifact image.
    #Thresholded artifact image
    fImg = (pImg - fImg).clip(thresh_art_min,thresh_art_max)

    #Azimuthal filtering in polar coordinates
    #dImg = medfilt(fImg, [azimuthal_kernel,1])
    dImg = polar_filter(fImg, azimuthal_kernel, 0)

    #Artifact Image transformed into cartesian coordinates
    ring_art = polar_to_cart(dImg,nX,nY, theta_step= 360.0/nThetas)



    if return_art:
        return img - ring_art, ring_art
    else:
        return img - ring_art
           
           
           