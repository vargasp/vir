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


def polyToParams3D(vec,printMe):

   # convert the polynomial form of the 3D-ellipsoid to parameters
   # center, axes, and transformation matrix
   # vec is the vector whose elements are the polynomial
   # coefficients A..J
   # returns (center, axes, rotation matrix)

   #Algebraic form: X.T * Amat * X --> polynomial form

   if printMe: print('\npolynomial\n',vec)

   Amat=np.array(
   [
   [ vec[0],     vec[3]/2.0, vec[4]/2.0, vec[6]/2.0 ],
   [ vec[3]/2.0, vec[1],     vec[5]/2.0, vec[7]/2.0 ],
   [ vec[4]/2.0, vec[5]/2.0, vec[2],     vec[8]/2.0 ],
   [ vec[6]/2.0, vec[7]/2.0, vec[8]/2.0, vec[9]     ]
   ])

   if printMe: print('\nAlgebraic form of polynomial\n',Amat)

   #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
   # equation 20 for the following method for finding the center
   A3=Amat[0:3,0:3]
   A3inv=np.linalg.inv(A3)
   ofs=vec[6:9]/2.0
   center=-np.dot(A3inv,ofs)
   if printMe: print ('\nCenter at:',center)

   # Center the ellipsoid at the origin
   Tofs=np.eye(4)
   Tofs[3,0:3]=center
   R = np.dot(Tofs,np.dot(Amat,Tofs.T))
   if printMe: print('\nAlgebraic form translated to center\n',R,'\n')

   R3=R[0:3,0:3]
   R3test=R3/R3[0,0]
   print('normed \n',R3test)
   s1=-R[3, 3]
   R3S=R3/s1
   (el,ec)=np.linalg.eig(R3S)

   recip=1.0/np.abs(el)
   axes=np.sqrt(recip)
   if printMe: print( '\nAxes are\n',axes  ,'\n')

   inve=np.linalg.inv(ec) #inverse is actually the transpose here
   if printMe: print('\nRotation matrix\n',inve)
   return (center,axes,inve)

