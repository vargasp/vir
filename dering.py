#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:11:51 2022

@author: vargasp
"""

import numpy as np
from skimage.transform import warp_polar
from scipy.signal import medfilt
from scipy.ndimage.interpolation import map_coordinates


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
    cart_data = map_coordinates(polar_data, coords, order=order, mode='constant', cval=np.nan)

    # The data is reshaped and returned
    return(cart_data.reshape(nY, nX))


def linear_kernel(k_max, width, steps=3):
    steps_max = int(np.ceil(k_max/2))
    steps = np.min([steps+1, steps_max])
    
    steps_filt = np.floor(np.linspace(1,steps,width,endpoint=False))

    return  2*np.floor(k_max*steps_filt/(steps-1)/2).astype(int) + 1


def polar_filter(img, kernel_max, axis):
    k_size = img.shape[1]
    
    kernel_filter = linear_kernel(kernel_max, k_size)
    
    kernel = [1,1]
    
    lf_img = np.zeros(img.shape)
    
    
    for k in np.unique(kernel_filter):
        print(k)
        
        kernel[axis] = k
        f_img = medfilt(img, kernel)
        
        idx = np.where(kernel_filter == k)[0]
        lf_img[:,idx] = f_img[:,idx]
        
    return lf_img


def dering(img,threshold_min=None,threshold_max=None,threshold_art=30,\
           azimuthal_kernel=11,radial_kernel=21, return_art=False):

    nX, nY = img.shape
    
    #Lower threshold for image segmentation
    if threshold_min is None:
        threshold_min = img.min()

    #Upper threshold for image segmentation
    if threshold_max is None:
        threshold_max = img.max()

    #Thresholded image
    tImg = img.clip(threshold_min,threshold_max)

    #Image transformed into polar coordinates
    nRadii = int(np.ceil(np.sqrt((nX/2)**2 + (nY/2)**2)))
    nThetas = 360
    pImg = warp_polar(tImg, output_shape=[nThetas,nRadii])

    #Radial median filtering in polar coordinates
    fImg = medfilt(pImg, [1,radial_kernel])
    
    #Subtract median imge to calculate artifactimge.
    #Thresholded artifact image
    fImg = (pImg - fImg).clip(0,threshold_art)

    #Azimuthal filtering in polar coordinates
    dImg = medfilt(fImg, [azimuthal_kernel,1])

    #Artifact Image transformed into cartesian coordinates
    ring_art = polar_to_cart(dImg,nX,nY)



    if return_art:
        return img - ring_art, ring_art
    else:
        return img - ring_art
           
           
           