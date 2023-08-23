# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np

import vir
import vir.psf as psf

from scipy.ndimage import rotate, map_coordinates, center_of_mass
from skimage.measure import EllipseModel
from scipy.spatial import ConvexHull

def fwhm_edge_profile(y, v=0.5):
    """
    Calcualtes the half max (or other) value of a one side of a point spread
    profile by linear interpolation.

    Parameters
    ----------
    y : (nY) numpy ndarray 
        One side of a psf. The profile must be normilized, with a maximum of
        1.0, a minimum of 0.0, and monotonically increasing or decreasing.
    v : int, optional
        The interpolated value. The dfault is 0.5 to give FWHM. Other values
        can be used to calculate FWTM, etc 

    Raises
    ------
    ValueError
        Raises ValueError if the profile is non-monotonic or if there are no 
        values less the 0.5 (cannot accurately interpolate width).

    Returns
    -------
    pixel_width : int
        The width in pixels between the value and the last pixel

    """
    
    
    #Checks and corrects for negaive gradient
    if y[0] > y[-1]:
        y = y[::-1]
    
    #Checks for monotonicity
    if np.all(np.diff(y) <= 0):
        raise ValueError("Non-monotonic Profile")
 
    #Checks for a value less than v to ensure a correct interpolation
    if y[0] > v:
        raise ValueError(f"No profile value less {v:.2f}")

    #Assigns an x axis based on pixel indices
    x = np.arange(y.size-1,-1,-1)
    
    #Returns the interpolated half width for the edge
    return np.interp(v,y,x)


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


def profile_img(img, p0, p1, dPix=1.0):
    """
    Returns a line profile from an image between to points p0 and p1

    Parameters
    ----------
    img : (nX, nY) or (nX, nY, nZ) numpy ndarray 
        Image array from which to extract profiles
    p0 : (..., nDims) numpy ndarray 
        The array of starting points for the profile. The final axis size must 
        be equal to the dimensions  of img
    p1 : (..., nDims) numpy ndarray 
        The array of ending points for the profile. The final axis size must 
        be equal to the dimensions  of img
    dPix : TYPE, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    profiles (..., nPix) numpy ndarray 
        An array of profiles from img 

    """
    
    #Swaps the last and first axis to have the correct format for map_coordinates
    p0 = np.moveaxis(np.array(p0),-1,0)
    p1 = np.moveaxis(np.array(p1),-1,0)

    #Calculates the unit vector(s) of the line profile between the two points   
    dPts = p1 - p0
    d = np.linalg.norm(dPts,axis=0)
    profile_vect = dPts/d

    #Calculates a uniform sampling vector for the profile
    nPix = int(d.max())
    sample_vect = np.linspace(0,nPix,int(nPix/dPix)+1)
    
    #Calculates the coordinates to sammple the gridded img
    coords = np.multiply.outer(profile_vect,sample_vect) + p0[...,np.newaxis]

    #Returns the interpolated values at coords
    return map_coordinates(img, coords, order=1)


def impulse_center(imp, method=0):

    if method == 0:
        return np.unravel_index(np.argmax(imp), imp.shape)
    
    if method == 1:
        imp /= imp.max()
        imp[imp < .5] = 0.0

        return center_of_mass(imp)
    
    


def imp_center(imp):
    #Determines the coordinats of higest value pixel
    C = np.array(np.unravel_index(np.argmax(imp), imp.shape))
    
    if imp.ndim == 2:
        imp_crop = imp[(C[0]-1):(C[0]+2), (C[1]-1):(C[1]+2)]    
    
    if imp.ndim == 3:
        imp_crop = imp[(C[0]-1):(C[0]+2), (C[1]-1):(C[1]+2), (C[2]-1):(C[2]+2)]

    #Determines the center of mass distance from highest value pixel
    dM = np.array(center_of_mass(imp_crop)) - 1

    if np.max(abs(dM)) >= 0.5:
        print("Warning!!! Center of mass pixel distnace exceed max pixel by 0.5 ")
    
    return tuple(C + dM) 


def impulse_profile(imp,samples=1024, angles=False):

    if imp.ndim == 2:
        #Dtermines the coordinats of higest value pixel
        #cX, cY = np.unravel_index(np.argmax(imp), imp.shape)
        cX, cY = imp_center(imp)
        
        
        #Calculates the x, y, and z coordinates around a sphere
        x,y = vir.sample_circle(radius=np.max(imp.shape), samples=samples)
        
        ret_angles = vir.cart2circ(x,y)[1]
        
        #Calculates profile starting and ending points for line profiles 
        p0 = np.tile([cX,cY],[samples,1])
        p1 = np.vstack([x,y]).T + np.array([cX,cY])

    if imp.ndim == 3:
        #Dtermines the coordinats of higest value pixel
        #cX, cY, cZ = np.unravel_index(np.argmax(imp), imp.shape)
        cX, cY, cZ = imp_center(imp)
        
        #Calculates the x, y, and z coordinates around a sphere
        x,y,z = vir.sample_sphere(radius=np.max(imp.shape), samples=samples)
        
        ret_angles = vir.cart2sph(x,y,z)[1:]
        
        #Calculates profile starting and ending points for line profiles 
        p0 = np.tile([cX,cY,cZ],[samples,1])
        p1 = np.vstack([x,y,z]).T + np.array([cX,cY,cZ])

    
    #Returns an array of line profiles spherialing emenating from the center
    #of the impulse

    imp_profiles = profile_img(imp, p0, p1)
    
    if angles:     
        return imp_profiles,ret_angles
    else:
        return imp_profiles
        

def impulse_characteristics(imp):
    samples = 128
    v = 0.5
    imp_profiles, angles = impulse_profile(imp,samples=samples, angles=True)
    theta, phi = angles
    
    fwhms = fwhm_edge_profiles(imp_profiles, v=v)
    
    xf, yf, zf = vir.sph2cart(fwhms, theta, phi)

    """
    #Calcuates the ellipse parameters
    lsvec = psf.ls_ellipsoid(xf, yf, zf)
    
    l,w,h = psf.polyToParams3D(lsvec,True)[1]
    
    vol = ConvexHull(np.array([xf,yf,zf]).T).volume
    area = ConvexHull(np.array([xf,yf,zf]).T).area
    return l, w, h, vol, area
    """
    return ConvexHull(np.array([xf,yf,zf]).T).area

def fwhm_edge_profiles(ys, v=0.5):
    samples = ys.shape[0]
    
    fwhms = np.empty(samples)
    
    for i in range(samples):
        fwhms[i] = fwhm_edge_profile(norm_profile(ys[i]),v=v)

    return fwhms

    
def norm_profile(profile,v=.5):
    
    #Normalizes profile
    profile /= profile.max()
   
    #Sets values after the impulse drops below 0 to 0
    #zero_ind = np.argwhere(imp_profiles<0)
    #zero_ind = zero_ind[np.searchsorted(zero_ind[:,0], np.arange(samples)),1]

    #for i in range(samples):
    #    imp_profiles[i,zero_ind[i]:] = 0.0

   
    
    #Sets all non monotonic values to 0
    non_mon = np.diff(profile,axis=-1)
    zero_ind = np.argwhere(non_mon > 0)
    
    if zero_ind.size != 0:
        zero_ind = zero_ind[0][0]
        
        if profile[zero_ind] > v:
            print(f"Warning: Lowest monotonic value ({profile[zero_ind]:.4f}) less than {v:.2f} ")
            
        profile[(zero_ind+1):] = 0.0


    return profile.clip(0.0)
    








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
    
def fwhm_pts(img, samples):
    
    angles = np.linspace(0,90,samples)
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
        
    return cX-x,y+cY



def fit_error_ellipse(x,y):
    ellipse = EllipseModel()
    ellipse.estimate(np.vstack([x,y]).T)

    return ellipse.params

def ellipse_params2xy(params, samples=500):
    xy = EllipseModel().predict_xy(np.linspace(0,2*np.pi,samples),params=params)

    return xy[:,0], xy[:,1]