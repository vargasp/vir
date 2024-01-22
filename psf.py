#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:33:57 2023

@author: pvargas21
"""

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.spatial import transform

import vir


def gaussianfunc1d(x, mu, sigma, A):
    """
    Returns the pdf of a 1d guassian distribution. Guassian paramters are
    scalar so the function can be used in model fitting algorithms. 

    Parameters
    ----------
    x : float or np.array
        The independent value(s) of the function.
    mu : float
        Mean value of the function.
    sigma : float
        Standard deviations of the function
    A : float
        The amplitude of the function

    Returns
    -------
    float or np.array
        The value of array of values of the pdf at x.
    """
    return A*np.exp(-(x - mu)**2/(2*sigma**2))


def gaussianfunc2d(xy, x_mu, y_mu, x_sigma, y_sigma, theta, A):
    """
    Returns the pdf of a 2d guassian distribution. This is a general case where
    the axes may not be aligned with the coordiante system. Guassian paramters
    are scalar so the function can be used in model fitting algorithms. 

    Parameters
    ----------
    xy : array of 2 floats or 2 np.arrays
        The independent value(s) x and y of the function.
    x_mu : TYPE
        Mean value in the x dimension of the function.
    y_mu : float
        Mean value in the y dimension of the function.
    x_sigma : float
        Standard deviations in the x dimension of the function
    y_sigma : float
        Standard deviations in the y dimension of the function
    theta : float
        Angle the axis along x_sigma makes with the positive x axis in radians
    A : float
        The amplitude of the function

    Returns
    -------
    float or np.array
        The value of array of values of the pdf at x and y.
    """
    
    #Creates np.arrays for elementwise arithmetic
    mu = np.array([x_mu, y_mu])
    sigma = np.array([x_sigma, y_sigma])

    #Translates the impulse to mean location       
    V = np.stack(xy, axis=-1) - mu

    #Generates the rotation matrix
    #scipy's transform's is 3x3, truncated to transform Nx2 matrices
    R = transform.Rotation.from_euler('z', theta).as_matrix()[:2,:2]
    
    #Creates the covariance matrix
    #OPTIMIZE This can probably be sped up with np.einsum notation
    COV = R @ np.diag((1.0/sigma)**2) @ R.T
    
    #Returns the gaussian pdf
    return A * np.exp(-0.5*((V @ COV) * V).sum(axis=-1))
        

def gaussianfunc3d(xyz, x_mu, y_mu, z_mu, \
                   x_sigma, y_sigma, z_sigma, \
                   theta, phi, psi, A):

    #Creates np.arrays for elementwise arithmetic
    mu = np.array([x_mu, y_mu, z_mu])
    sigma = np.array([x_sigma, y_sigma, z_sigma])

    #Translates the impulse to mean location       
    V =  np.stack(xyz, axis=-1) - mu
       
    #Generates the rotation matrix    
    R = transform.Rotation.from_euler('xyz', [theta,phi,psi]).as_matrix()

    #Creates the covariance matrix
    #OPTIMIZE This can probably be sped up with np.einsum notation
    COV = R @ np.diag((1.0/sigma)**2) @ R.T
    
    #Returns the gaussian pdf    
    return A * np.exp(-0.5*((V @ COV) * V).sum(axis=-1))


def gaussianfunc2d_ortho(xy, x_mu, y_mu, x_sigma, y_sigma, A):
    """
    Returns the pdf of a 2d guassian distribution. This is a special case where
    the axes are aligned with the coordiante system. Guassian paramters are
    scalar so the function can be used in model fitting algorithms. 
    
    Parameters
    ----------
    xy : array of 2 floats or 2 np.arrays
        The independent value(s) x and y of the function.
    x_mu : TYPE
        Mean value in the x dimension of the function.
    y_mu : float
        Mean value in the y dimension of the function.
    x_sigma : float
        Standard deviations in the x dimension of the function
    y_sigma : float
        Standard deviations in the y dimension of the function
    A : float
        The amplitude of the function

    Returns
    -------
    float or np.array
        The value of array of values of the pdf at x and y.
    """
    x,y = xy
    
    return A*np.exp(-(x - x_mu)**2/(2*x_sigma**2) \
                    -(y - y_mu)**2/(2*y_sigma**2))


def gaussianfunc3d_orth0(xyz, x_mu, y_mu, z_mu, x_sigma, y_sigma, z_sigma, A):
    """
    Returns the pdf of a 3d guassian distribution. This is a special case where
    the axes are aligned with the coordiante system. Guassian paramters are
    scalar so the function can be used in model fitting algorithms. 
    
    Parameters
    ----------
    xyz : array of 3 floats or 3 np.arrays
        The independent value(s) x, y  and z of the function.
    x_mu : TYPE
        Mean value in the x dimension of the function.
    y_mu : float
        Mean value in the y dimension of the function.
    z_mu : float
        Mean value in the z dimension of the function.
    x_sigma : float
        Standard deviations in the x dimension of the function
    y_sigma : float
        Standard deviations in the y dimension of the function
    z_sigma : float
        Standard deviations in the z dimension of the function
    A : float
        The amplitude of the function

    Returns
    -------
    float or np.array
        The value of array of values of the pdf at x, y and z.
    """
    x,y,z = xyz         

    return A*np.exp(-(x - x_mu)**2/(2*x_sigma**2) \
                    -(y - y_mu)**2/(2*y_sigma**2) \
                    -(z - z_mu)**2/(2*z_sigma**2))


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
    return gaussianfunc1d(x, mu, sigma, A)


def gaussian2d(mus=(0.0,0.0), sigmas=(5.0,5.0), theta=0, A=None, \
               nX=128, nY=128):
    
    x_mu,y_mu = mus
    x_sigma, y_sigma = sigmas
    
    x = vir.censpace(nX)
    y = vir.censpace(nY)
    x, y = np.meshgrid(x,y)
    
    if A is None:
        A = 1.0 / (x_sigma*y_sigma * (2*np.pi))

    return gaussianfunc2d((x,y), x_mu, y_mu, x_sigma, y_sigma, theta, A)


def gaussian3d(mus=(0.0,0.0,0.0), sigmas=(5.0,5.0,5.0), angles=(0.0,0.0,0.0), \
               A=None, nX=128, nY=128, nZ=128):
    
    x_mu, y_mu, z_mu = mus
    x_sigma, y_sigma, z_sigma = sigmas
    theta, phi, psi = angles
    
    x = vir.censpace(nX)
    y = vir.censpace(nY)
    z = vir.censpace(nZ)
    x, y, z = np.meshgrid(x,y,z)
    
    if A is None:
        A = 1.0 / (x_sigma*y_sigma*z_sigma * (2*np.pi)**(3/2))
            
    return gaussianfunc3d((x,y,z), x_mu, y_mu, z_mu, x_sigma, y_sigma, z_sigma,
                          theta, phi, psi, A)
        

def moments(imp):
    """
    Calculates the moments of an impulse by maximum, center of mass, and RMSE.
    Used to estimate intital guesses for a guassian model fitting

    Parameters
    ----------
    imp : 1, 2, or 3d np.array
        An impusle with values on a rectllinear grid of equal sizes

    Returns
    -------
    CoMs : scalar or array
        The estimated mu of the impulse.
    sigmas : scalar or array
        The estimated sigma of the impulse.
    A : scalar
        The estimated amplitude of the impulse.
    """
    
    #Estimates the amplitude
    A = imp.max()
    
    #Estiamtes mu valeues by the center of mass
    CoMs  = ndimage.center_of_mass(imp)

    #Creates orthogonal profiles to estimate sigma
    profiles = []
    if len(CoMs) == 1:
        profiles.append(imp)
    
    elif len(CoMs) == 2:
        profiles.append(imp[:, int(CoMs[1])])
        profiles.append(imp[int(CoMs[0]), :])

    elif len(CoMs) == 3:
        profiles.append(imp[:, int(CoMs[1]), int(CoMs[2])])
        profiles.append(imp[int(CoMs[0]), :, int(CoMs[2])])
        profiles.append(imp[int(CoMs[0]), int(CoMs[1]), :])

    sigmas = ()
    for i in range(len(CoMs)):
        #Estimate sigma by the root mean squared deviation
        sigmas += (np.sqrt(np.abs((np.arange(profiles[i].size)-CoMs[i])**2*profiles[i]).sum()/profiles[i].sum()),)

    #Returns the gaussian parameter estimates
    return CoMs, sigmas, A


def fitGaussian2d(x, y, imp):
    """
    Calculates the parameters of a 2d guassian.

    Parameters
    ----------
    x : nX np.array
        The x coordinates of a 2d guassian impulse 
    y : nY np.array
        The y coordinates of a 2d guassian impulse 
    imp : (nX, nY) np.array 
        The impulse pdf.

    Returns
    -------
    params : array
        The guassian parameters of the impulse
        (x_mu, y_mu, x_sigma, y_sigma, theta, A)

    """
    #Estimates the intial gausian paramters
    (mu_x0, mu_y0), (sigma_x0, sigma_y0), A = moments(imp)
    p0  = [y[int(mu_y0)], x[int(mu_x0)], sigma_y0, sigma_x0, 0.0, A]
    
    #Unravels the arrays to be used in curvefit
    x2, y2 = np.meshgrid(x,y)
    x2 = np.ravel(x2)
    y2 = np.ravel(y2)
    imp = np.ravel(imp)

    #Estimates the parameters using least squares
    #OPTIMIZE This can probably be improved with a different optimization alg
    return curve_fit(gaussianfunc2d, np.array((x2,y2)), imp, p0=p0)
    

def fitGaussian3d(x, y, z, imp):
    """
    Calculates the parameters of a 3d guassian.

    Parameters
    ----------
    x : nX np.array
        The x coordinates of a 3d guassian impulse 
    y : nY np.array
        The y coordinates of a 3d guassian impulse 
    z : nZ np.array
        The z coordinates of a 3d guassian impulse 
    imp : (nX, nY, nZ) np.array 
        The impulse pdf.

    Returns
    -------
    params : array
        The guassian parameters of the impulse
        (x_mu, y_mu, z_mu, x_sigma, y_sigma, z_sigma, theta, phi, psi, A)

    """
    
    #Estimates the intial gausian paramters
    (mu_x0, mu_y0, mu_z0), (sigma_x0, sigma_y0,sigma_z0), A = moments(imp)    
    p0  = [y[int(mu_y0)], x[int(mu_x0)], z[int(mu_z0)], \
           sigma_y0, sigma_x0, sigma_z0, \
           0.0, 0.0, 0.0, A]
    
    #Unravels the arrays to be used in curvefit
    x2, y2, z2 = np.meshgrid(x,y,z)
    x2 = np.ravel(x2)
    y2 = np.ravel(y2)
    z2 = np.ravel(z2)
    imp = np.ravel(imp)

    #Estimates the parameters using least squares
    #OPTIMIZE This can probably be improved with a different optimization alg
    
    """
    TESTING
    bounds =((x.min(),y.min(),z.min(), 0.0,0.0,0.0, 0.0,0.0,0.0, A*0.5 ),\
             (x.max(),y.max(),z.max(), \
              x.max()-x.min(),y.max()-y.min(),z.max()-z.min(), \
              2.0*np.pi,2.0*np.pi,2.0*np.pi, A*1.5 ))
    par0 = curve_fit(gaussianfunc3d, np.array((x2,y2,z2)), imp, p0=p0, bounds=bounds, full_output=True)
    
    par1 = curve_fit(gaussianfunc3d, np.array((x2,y2,z2)), imp, p0=p0, full_output=True)
    
    print(par0)
    print(par1)
    """
    return curve_fit(gaussianfunc3d, np.array((x2,y2,z2)), imp, p0=p0)


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


def estimatePSF(im_t, im_b, mask=None, kernel_size=None):
    fft_im_t = np.fft.fftshift(np.fft.fft2(im_t))
    fft_im_b = np.fft.fftshift(np.fft.fft2(im_b))
    fft_psf = fft_im_b / fft_im_t
    
    if mask is not None:
        fft_psf = vir.mask(fft_psf, mask)
    
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fft_psf)))
    psf = np.real(psf)
    
    if kernel_size is None:
        return psf
    else:
        k0 = int(im_t.shape[0]/2 - kernel_size[0]/2)
        k1 = int(im_t.shape[1]/2 - kernel_size[1]/2)

        return psf[k0:(k0+kernel_size[0]),k1:(k1+kernel_size[1])]







#least squares fit to a 3D-ellipsoid
#  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1
#
# Note that sometimes it is expressed as a solution to
#  Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
# where the last six terms have a factor of 2 in them
# This is in anticipation of forming a matrix with the polynomial coefficients.
# Those terms with factors of 2 are all off diagonal elements.  These contribute
# two terms when multiplied out (symmetric) so would need to be divided by two

def ls_ellipsoid(xx,yy,zz):

   # change xx from vector of length N to Nx1 matrix so we can use hstack
   x = xx[:,np.newaxis]
   y = yy[:,np.newaxis]
   z = zz[:,np.newaxis]

   #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
   J = np.hstack((x*x,y*y,z*z,x*y,x*z,y*z, x, y, z))
   K = np.ones_like(x) #column of ones

   #np.hstack performs a loop over all samples and creates
   #a row in J for each x,y,z sample:
   # J[ix,0] = x[ix]*x[ix]
   # J[ix,1] = y[ix]*y[ix]
   # etc.

   JT=J.transpose()
   JTJ = np.dot(JT,J)
   InvJTJ=np.linalg.inv(JTJ);
   ABC= np.dot(InvJTJ, np.dot(JT,K))

# Rearrange, move the 1 to the other side
#  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
#    or
#  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
#  where J = -1
   eansa=np.append(ABC,-1)

   return (eansa)


def polyToParams3D(vec,printMe=False):

   # convert the polynomial form of the 3D-ellipsoid to parameters
   # center, axes, and transformation matrix
   # vec is the vector whose elements are the polynomial
   # coefficients A..J
   # returns (center, axes, rotation matrix)

   #Algebraic form: X.T * Amat * X --> polynomial form

   #if printMe: print('\npolynomial\n',vec)

   Amat=np.array(
   [
   [ vec[0],     vec[3]/2.0, vec[4]/2.0, vec[6]/2.0 ],
   [ vec[3]/2.0, vec[1],     vec[5]/2.0, vec[7]/2.0 ],
   [ vec[4]/2.0, vec[5]/2.0, vec[2],     vec[8]/2.0 ],
   [ vec[6]/2.0, vec[7]/2.0, vec[8]/2.0, vec[9]     ]
   ])

   #if printMe: print('\nAlgebraic form of polynomial\n',Amat)

   #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
   # equation 20 for the following method for finding the center
   A3=Amat[0:3,0:3]
   A3inv=np.linalg.inv(A3)
   ofs=vec[6:9]/2.0
   center=-np.dot(A3inv,ofs)
   #if printMe: print ('\nCenter at:',center)

   # Center the ellipsoid at the origin
   Tofs=np.eye(4)
   Tofs[3,0:3]=center
   R = np.dot(Tofs,np.dot(Amat,Tofs.T))
   #if printMe: print('\nAlgebraic form translated to center\n',R,'\n')

   R3=R[0:3,0:3]
   R3test=R3/R3[0,0]
   #if printMe: print('normed \n',R3test)
   s1=-R[3, 3]
   R3S=R3/s1
   (el,ec)=np.linalg.eig(R3S)

   recip=1.0/np.abs(el)
   axes=np.sqrt(recip)
   #if printMe: print( '\nAxes are\n',axes  ,'\n')

   inve=np.linalg.inv(ec) #inverse is actually the transpose here
   #if printMe: print('\nRotation matrix\n',inve)
   return (center,axes,inve)




"""
!!!Deprecated functions!!!


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


def gaussian2d(mu=(0.0,0.0), sigma=(5.0,5.0), theta=0, A=None, \
               nX=128, nY=128):
    
    X = vir.censpace(nX)
    Y = vir.censpace(nY)

    mu = np.array(mu)
    sigma = np.array(sigma)
       

    c, s = np.cos(theta), np.sin(theta)
    R  = np.array([[c, -s], [s, c]])
    
    #R = transform.Rotation.from_euler('z', theta, degrees=True).as_matrix()[:2,:2]
    
    COV = R @ np.diag((1/sigma)**2) @ R.T
    
    V2 = np.dstack(np.meshgrid(X,Y))
    
    if A is None:
        A = 1.0 / (np.prod(sigma) * (2*np.pi))
    
    return A * np.exp(-0.5*((V2 @ COV) * V2).sum(axis=2))


def gaussian3d(mu=(0.0,0.0,0.0), sigma=(5.0,5.0,5.0), angles=(0.0,0.0,0.0),\
               A=None, nX=128, nY=128, nZ=128):
    
    X = vir.censpace(nX)
    Y = vir.censpace(nY)
    Z = vir.censpace(nZ)

    mu = np.array(mu)
    sigma = np.array(sigma)
       
    R = transform.Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    COV = R @ np.diag((1/sigma)**2) @ R.T
    
    V2 = np.stack(np.meshgrid(X,Y,Z), axis=3)
    
    if A is None:
        A = 1.0 / (np.prod(sigma) * (2*np.pi)**(3/2))
    
    return A * np.exp(-0.5*((V2 @ COV) * V2).sum(axis=3))


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
        

"""




