# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
from scipy.ndimage import rotate
from skimage.measure import EllipseModel

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
    
    
    
def fwhm_1d(profile, allow_multiple_max=True):
    """
    Calculates the full width half max of a psf in line profile. 

    Parameters
    ----------
    profile : numpy ndarray 
        Line profile of a psf. The profile must have adjacent maximum values,
        be monotonically increasing to the maximum, and monotonically
        decreasing after the maximum.
        
    allow_multiple_max : bool 
        If True the profile can contain more than one maximum value. However,
        all of these values must be adjacent pixels. Default is True

    Raises
    ------
    ValueError
        Raises ValueError if more than one maximum values is in the profile, 
        and allow_multiple_max is False
        Raises ValueError if the profile is multimodal

    Returns
    -------
    fwhm : int
        The full width half max in pixels of the profile

    """
    
    #Normalizes profile to a maximim of 1.0
    profile = profile/np.max(profile)
    
    #Determines the number of maximim values
    if np.count_nonzero(profile == profile.max()) > 1:
        
        if allow_multiple_max:
            
            #Determines the maximum value indices
            maxes = np.flatnonzero(profile == np.max(profile))
            
            #Confirms max values are adjacent
            if np.all(np.diff(maxes) != 1):
                raise ValueError("Profile is multimodal")
            
            #Determines maximum indices
            cl = maxes[0]
            cr = maxes[-1]

        else:
            raise ValueError("Profile contains more than one maximum value.")
            
    else:
        #Determines maximum index for one maximum
        cl = np.argmax(profile)
        cr = cl
    
    #Calcualtes left side width at half max
    l = fwhm_edge_profile(profile[:(cl+1)])

    #Calcualtes right side width at half max
    r = fwhm_edge_profile(profile[cr:][::-1])
    
    #Returns fwhm (left width + right width + width between maximum points)
    return l+r + (cr - cl)


def fwhm_orth(img, allow_multiple_max=True):
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
    

def fwhm_pts(img):
    
    angles = np.linspace(0,90,50)
    img_pad = pad_image(img)
    cX, cY = np.unravel_index(np.argmax(img_pad), img_pad.shape)

    x = np.zeros(angles.size*4)
    y = np.zeros(angles.size*4)
    for i, angle in enumerate(angles):
        img_rot = rotate(img_pad, angle, reshape=False, order=1)
        img_rot = img_rot/img_rot.max()

        r = fwhm_edge_profile(img_rot[cX:,cY][::-1])
        x[i*4] = r*np.cos(angle*np.pi/180)
        y[i*4] = r*np.sin(angle*np.pi/180)
        
        
        r = fwhm_edge_profile(img_rot[:(cX+1),cY])
        x[i*4+1] = r*np.cos((angle+180)*np.pi/180)
        y[i*4+1] = r*np.sin((angle+180)*np.pi/180)
        
        
        r = fwhm_edge_profile(img_rot[cX,cY:][::-1])
        x[i*4+2] = r*np.cos((angle+90)*np.pi/180)
        y[i*4+2] = r*np.sin((angle+90)*np.pi/180)
        
        r = fwhm_edge_profile(img_rot[cX,:(cY+1)])
        x[i*4+3] = r*np.cos((angle-90)*np.pi/180)
        y[i*4+3] = r*np.sin((angle-90)*np.pi/180)
        
    return x+cX,y+cY


def fit_error_ellipse(x,y):
    ellipse = EllipseModel()
    ellipse.estimate(np.vstack([x,y]).T)

    return ellipse.params

def ellipse_params2xy(params, samples=500):
    xy = EllipseModel().predict_xy(np.linspace(0,2*np.pi,samples),params=params)

    return xy[:,0], xy[:,1]
