# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import rotate

def pad_vals(n, c):
    if n % 2:
        pad = int(2*(np.floor(n/2) - c))
    else:
        pad = int(2*((n-1)/2 - c))

    return [max(pad,0), -min(pad,0)]


def fwhm_edge_profile(y):
    """
    Calcualtes the half max value of a one side of a point spread profile by
    linear interpolation.

    Parameters
    ----------
    y : (nY) numpy ndarray 
        One side of a psf. The profile must be normilized, with a maximum of
        1.0, a minimum of 0.0, and monotonically increasing. The last value in
        the array should be 1.0.

    Raises
    ------
    ValueError
        Raises ValueError if the profile is non-monotonic or if there are no 
        values less the 0.5 (cannot accurately interpolate width).

    Returns
    -------
    pixel_width : int
        The width in pixels between the last value and 0.5

    """
    
    #Checks for monotonicity
    if np.all(np.diff(y) <= 0):
        raise ValueError("Non-monotonic Profile")
 
    #Checks for a value less than 0.5 to ensure a correct interpol;ation
    if y[0] > 0.5:
        raise ValueError("No profile value less 0.5")
    
    #Assigns an x axis based on pixel indices
    x = np.arange(y.size-1,-1,-1)
    
    #Returns the interpolated half width for the edge
    return np.interp(.5,y,x)
    
    
    
def fwhm_1d(profile):
    """
    Calculates the full width half max of a psf in line profile. 

    Parameters
    ----------
    profile : numpy ndarray 
        Line profile of a psf. The profile must have only one maximum value, be
        monotonically increasing to the maximum, and monotonically decreasing
        after the maximum.

    Raises
    ------
    ValueError
        Raises ValueError if more than one maximum values is in the profile.

    Returns
    -------
    fwhm : int
        The full width half max in pixels of the profile

    """
    
    #Normalizes profile to a maximim of 1.0
    profile = profile/np.max(profile)
    
    #Confirms only one maximim value exists
    if np.count_nonzero(profile == profile.max()) > 1:
        raise ValueError("Profile contains more than one maximum value.")
    
    #Determines maximum index
    c = np.argmax(profile)
    
    #Calcualtes left side width at half max
    l = fwhm_edge_profile(profile[:(c+1)])

    #Calcualtes right side width at half max
    r = fwhm_edge_profile(profile[c:][::-1])
    
    #Returns fwhm
    return l+r


def fwhm_orth(img):
    """
    Calculates the orthagonal full width half max values of a psf in an image.

    Parameters
    ----------
    img : (nX, nY) numpy ndarray 
        2d arrary with a PSF located at the maximum value pixel

    Raises
    ------
    ValueError
        Raises ValueError if more than one maximum values is in the image.

    Returns
    -------
    fwhmX : TYPE
        Full with half max of the PSF in the x dimension.
    fwhmY : TYPE
        Full with half max of the PSF in the y dimension.

    """
    
    #Confirms only one maximim value exists
    if np.count_nonzero(img == img.max()) > 1:
        raise ValueError("Img contains more than one maximum value.")
    
    if img.ndim == 1:
        return fwhm_1d(img)
    
    elif img.ndim == 2:        
        #Determines maximum indices
        cX, cY = np.unravel_index(np.argmax(img), img.shape)
        
        #Calcualtes x and yaxis full width half max
        return fwhm_1d(img[:,cY]), fwhm_1d(img[cX,:])

    elif img.ndim == 3:        
        #Determines maximum indices
        cX, cY, cZ = np.unravel_index(np.argmax(img), img.shape)
        
        #Calcualtes x and yaxis full width half max
        return fwhm_1d(img[:,cY,cZ]), fwhm_1d(img[cX,:,cZ]), fwhm_1d(img[cX,cY,:])
    else:
        raise ValueError("Img must be between 1 and 3 dimesions")


def fwhm_ang(img, angle):
    nX, nY = img.shape
    
    cX, cY = np.unravel_index(np.argmax(img), img.shape)
    
    padX = pad_vals(nX, cX)
    padY = pad_vals(nY, cY)
    
    img_pad = np.pad(img, [padX, padY])
    cX, cY = np.unravel_index(np.argmax(img_pad), img_pad.shape)
    
    img_rot = rotate(img_pad, angle, reshape=False, order=1)
    return fwhm_orth(img_rot)
    


nX = 15
nY = 15
center = (8,7)

x,y = np.indices((nX,nY))
    
image = -np.sqrt((x - center[0])**2 + (y - center[1])**2)
image -= image.min()
image = image/image.max()
print(fwhm_orth(image))
print(fwhm_ang(image,90))

