# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
from scipy.ndimage import rotate


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

def pad_vals(n, c):
    if n % 2:
        pad = int(2*(np.floor(n/2) - c))
    else:
        pad = int(2*((n-1)/2 - c))

    return [max(pad,0), -min(pad,0)]


def pad_image(img):
    
    if img.ndim == 2:
        nX, nY = img.shape
    
        cX, cY = np.unravel_index(np.argmax(img), img.shape)
        
        padX = pad_vals(nX, cX)
        padY = pad_vals(nY, cY)
        
        return np.pad(img, [padX, padY])

    elif img.ndim == 3:
        nX, nY, nZ = img.shape
    
        cX, cY, cZ = np.unravel_index(np.argmax(img), img.shape)
        
        padX = pad_vals(nX, cX)
        padY = pad_vals(nY, cY)
        padZ = pad_vals(nZ, cZ)
        
        return np.pad(img, [padX, padY, padZ])
    else:
        raise ValueError("Img must be between 2 and 3 dimesions")


def fwhm_ang(img, angles):
    
    img_pad = pad_image(img)
    
    if img_pad.ndim == 2:
        angles = np.array(angles, ndmin=1)
        cX, cY = np.unravel_index(np.argmax(img_pad), img_pad.shape)
        
        fwhms = np.zeros((angles.size, 2))
        for i, angle in enumerate(angles):
            img_rot = rotate(img_pad, angle, reshape=False, order=1)
            fwhms[i,:] = fwhm_orth(img_rot)

        return fwhms

    elif img_pad.ndim == 3:
        angles = np.array(angles, ndmin=2)
        cX, cY, cZ = np.unravel_index(np.argmax(img_pad), img_pad.shape)
        
        fwhms = np.zeros((angles.shape[0], 3))
        for i, angle in enumerate(angles):
            img_rot = rotate(img_pad, angle[0], axes=(1,2), reshape=False, order=1)
            img_rot = rotate(img_rot, angle[1], axes=(0,2), reshape=False, order=1)
            img_rot = rotate(img_rot, angle[2], axes=(0,1), reshape=False, order=1)
            fwhms[i, :] = fwhm_orth(img_rot)
        
        return fwhms.squeeze()



def gaussian2d(A, mus, sigmas, theta=0, nX=128, nY=128):
    
    x_mu,y_mu = mus
    x_sigma, y_sigma = sigmas
    
    x,y = np.indices((nX,nY))
    
    a = (np.cos(theta)**2 / (2*x_sigma**2) + np.sin(theta)**2 / (2*y_sigma**2))
    b = (np.sin(2*theta)**2 / (2*x_sigma**2) - np.sin(2*theta)**2 / (2*y_sigma**2))
    c = (np.sin(theta)**2 / (2*x_sigma**2) + np.cos(theta)**2 / (2*y_sigma**2))
    
    k =  A * np.exp(-a*(x - x_mu)**2 - b*(x - x_mu)*(y - y_mu) - c*(y - y_mu)**2)
    
    return k
    


def gaussian3d(A,mus,sigmas, nX=128, nY=128, nZ=128):
    
    x_mu, y_mu, z_mu = mus
    x_sigma, y_sigma, z_sigma = sigmas
    
    x,y,z = np.indices((nX,nY,nZ))
    
    N = 1/ (x_sigma*y_sigma*z_sigma * (2*np.pi)**(3/2))
    k =  A * np.exp(-(x - x_mu)**2/x_sigma**2 - (y - y_mu)**2/y_sigma**2 - (z - z_mu)**2/z_sigma**2)
    
    return k