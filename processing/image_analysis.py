import pylab as py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn


def centroid(array, dPixel=1.0, center=True):
    """
    Calculates and returns the centroid for and array
        
    Parameters:
        array:  an N-d array or list
        size:   pixel dimensions
            
    Returns:
        a tuple of the centroid dimensions
    """

    #Converts array and dPixel to numpy arrays
    array = np.array(array, dtype=float)
    dPixel = np.array(dPixel, dtype=float)

    #Determines the dimensions and elements in the array
    Dims = array.shape
    nDims = len(Dims)
    
    #Creates defaut size array
    if(nDims != 1 and dPixel.size == 1):
        dPixel = np.repeat(dPixel, nDims)
    
    #Error control
    if (len(array.shape) != dPixel.size):
        print('Array and size dimensions do not match')
        return 0
    
    if(center == True):
        center = dPixel*Dims/2.0
    else:
        center = dPixel*0.0

    #Calculate centroids
    if nDims ==1:
        vect = dPixel*np.linspace(0.5, Dims[0]-0.5,Dims[0])
        return np.average(vect, weights=array) - center[0]

    coords = ()
    for i in range(nDims):
        vect = dPixel[i]*np.linspace(0.5, Dims[i]-0.5,Dims[i])
        q = np.setdiff1d(range(nDims), [i])
        w = np.sum(array, axis=q)
        coords += (np.average(vect, weights = w) - center[i],)
            
    return coords


def mask_circle(img_shape=(512,512), center=None, radius=10):
    """
    Returns a boolean array with a cicular mask with the specifiec paramters
        
    Parameters:
    
        center:    The location of the center of the mask (j,i)  [index]
        radius:    The size of the mask's radius in pixels
        img_shape: List or tuple specifying pixels [nrows,ncols] for the image
       
        Returns:
        
        A boolean array with a cicular mask
    """

    if center == None:
        center = ()
        center += (img_shape[0]/2,)
        center += (img_shape[1]/2,)
    
    x,y = np.indices((img_shape[0],img_shape[1]))
    
    return (x - center[0])**2 + (y - center[1])**2 < radius**2


def fwhm_xy(image, dPixel = 1.0):
    """
    Calculates and returns the fwhm for an array in the xy directions
        
    Parameters:
        array:  a 2d array
        size:   pixel dimensions
            
    Returns:
        a tuple of the FWHM calcualtions 
    """
    image_max = np.max(image)
    image_half_max = image_max/2.0
    
    coords = np.argwhere(image == image_max)[0]
    
    profiles = (image[coords[0],:], image[:,coords[1]])

    fwhm = ()
    for profile in profiles:
        p = np.where(profile.clip(0,image_half_max) == image_half_max)
        l_index = p[0][0]
        r_index = p[0][len(p[0]) - 1]

        slope_l = profile[l_index] - profile[l_index - 1]
        x_l = l_index - (profile[l_index] - image_max*.5)/slope_l
        slope_r = profile[r_index + 1] - profile[r_index]
        x_r = r_index - (profile[r_index] - image_max*.5)/slope_r 
    
        fwhm += (dPixel*(x_r - x_l), )

    return fwhm


def fwhm_profile(profile):
    profile_min = np.max(profile)
    profile_max = np.max(profile)
    profile_half_max = profile_max/2.0
    
    ind = np.where(profile.clip(profile_min,profile_half_max) == profile_half_max) 
    
    

def center_image_max(image):
    """
    """
    image_max = np.max(image)    
    coords = np.argwhere(image == image_max)[0]
    
    roll_image = image
    roll_image = np.roll(roll_image, image[0].size/2 - coords[0], axis=0)
    roll_image = np.roll(roll_image, image[1].size/2 - coords[1], axis=1)

    return roll_image


def roll_nd(array, shift):
    """
    """
    dims = len(array.shape)

    if(len(shift) != array):
        print("Dimension Error")
        exit

    r_array = array.copy()    
    for i in range(dims):
        r_array = np.roll(r_array, shift[i], axis=0)

    return r_array
    
    
    

def mtf3(image, dPixel=1.0, sigmas=3.0, nFreq=512, offset=True):
    '''
    Calculates the MTF of an impulse via 2D FFT
    MTF is averaged across
    
    image = 2D image of an impulse with ij indexing
    '''

    nX, nY = image.shape
    
    #Calculate the radius of the ROI surrounding the impulse
    ROI_r = sigmas*np.max(fwhm_xy(image, dPixel = dPixel)) \
        /(2*np.sqrt(2*np.log(2)))/dPixel

    #Calculate the center of the impulse
    ROI_c = np.argwhere(image  == np.max(image))[0]
    
    #Calculate the ROI around the center of the impluse
    ROI = mask_circle(img_shape=(nX,nY), center=ROI_c, radius=ROI_r)    

    #Subtract the offset
    if offset == True:
        image = image - np.mean(image*~ROI)
        print(np.mean(image*~ROI))
    else:
        image = image - offset
        print(offset)

            
    #Calculate the normalized 2D MTF
    mtf = np.fft.fftshift(np.abs(np.fft.fft2(image,(nFreq,nFreq)) ))
    mtf = mtf/mtf[nFreq/2, nFreq/2]
    
    #Interpolate at discrete frequencies
    mtf1d = np.zeros([nFreq/2])
    mtf1d[0] = mtf[nFreq/2, nFreq/2]
    
    for i in range(1,nFreq/2):
        mtf1d[i] = interp_polar_values(mtf, i, origin=(nFreq/2, nFreq/2)).mean()
    
    freq = np.fft.fftfreq(nFreq, d = dPixel)
    
    return (mtf1d, freq[0:nFreq/2])
    
    
    
def interp_polar_values(image, radius, origin=None, nSamples=100, extent=2*np.pi, theta0=0):
    '''
    Interpolates values on an image the fall on a circle 
    '''
    nX, nY = image.shape
    
    if origin == None:
        origin = (nX/2.0 - .5, nY/2.0 - .5)
    
    X = np.linspace(0,nX-1,nX)
    Y = np.linspace(0,nY-1,nY)

    Thetas = np.linspace(0,extent - extent/nSamples, nSamples) + theta0

    x = radius*np.cos(Thetas) + origin[0]
    y = radius*np.sin(Thetas) + origin[1]

    return interpn((X,Y), image, np.array([x,y]).T)

    
def plot_mtf(image, dPixel=1.0):
    mtf = mtf3(image, dPixel=dPixel)
    
    fig = plt.figure()
    
    plt.plot(mtf[1],mtf[0])
    
    plt.title('Local MTF', fontsize=14)
    plt.ylabel('MTF(f)', fontsize=12)
    plt.xlabel('Spatial Frequency line pairs/mm', fontsize=12)
    plt.xlim([0,2])

    
def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)
    
    