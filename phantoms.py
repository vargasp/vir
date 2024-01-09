#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:22:55 2020

@author: vargasp
"""

import os
import numpy as np
import vir

class Phantom:
    """
    Attributes
    ----------
    nS: int
        The number of spheres in the phantom.
    S : (nS, 5) numpy ndarray
        A 2d array of spheres with the parmeters [x,y,z,r,value]
    
    Methods
    -------
    updateSpheres(self, spheres)
        Updates the array of projection angles with a custom array
    formatSpheres(spheres)
        Formats the array_like object to (nS, 5) numpy ndarray     
    sphereExtent()
        Returns the min and max extent of the phantom
            
    Raises
    ------
    ValueError
        If spheres array does not have the correct shape.        
    """
    def __init__(self,spheres=None):
        """
        Parameters
        ----------
        spheres : (nS, 5) array_like
            The number of spheres in the phantom. Default is None
        """
        if spheres is not None:
            self.S = self.formatSpheres(spheres)
                        
        else:
            self.S = np.empty([0,5])
            
        self.nS = self.S.shape[0]    


    def updateSpheres(self,spheres):
        """
        updateSpheres(self, spheres)
            Updates the array of projection angles with a custom array

        Parameters
        ----------
        spheres : (nS, 5) array_like
            The number of spheres in the phantom. Default is None
        """
        self.S = self.formatSpheres(spheres)
        self.nS = self.S.shape[0]


    def formatSpheres(self, spheres):
        """
        formatSpheres(spheres)
            Formats the array_like object to (nS, 5) numpy ndarray

        Parameters
        ----------
        spheres : (nS, 5) array_like
            The number of spheres in the phantom. Default is None
        """
        spheres = np.array(spheres, copy=True, dtype=float)
        
        if spheres.shape[-1] != 5:
            raise ValueError('Sphere parameters must have (nS, 5) shape')
            
        if spheres.ndim == 1:
            spheres = spheres[np.newaxis,:]
            
        return spheres
    
    def sphereExtent(self):
        """
        sphereExtent()
            Returns the min and max extent of the phantom

        Returns
        -------
         : ((3), (3)) tuple
            The min and max paramters of the phantom (x,y,z) 
        """
        min_values = np.min(self.S[:,:3].T - self.S[:,3],axis=1)
        max_values = np.max(self.S[:,:3].T + self.S[:,3],axis=1)

        return (min_values, max_values)
    
    def minDetectorElems(self, dDets, pow2=True): 
        dDets = np.array(dDets,dtype=float)
        if dDets.size == 1:
            dDets = np.repeat(dDets,2)
        
        min_values, max_values  = self.sphereExtent()
        dValues =  (max_values - min_values)
        
        nW = np.ceil(np.max(dValues[:2])/dDets[0])
        nH = np.ceil(dValues[2])/dDets[1]
        
        if pow2:
            return np.array((2**np.ceil(np.log2(nW)), \
                            2**np.ceil(np.log2(nH))), dtype=int)
        else:
            return np.array((nW,nH), dtype=int)


class DiscretePhantom:
    """
    Attributes
    ----------
    nX : int
        The number of samples in the X dimension of the image space.
    nY : int
        The number of samples in the Y dimension of the image space.
    nZ : int
        The number of samples in the Z dimension of the image space.
    dX : float
        The sampling interval in the X dimension of the image space.
    dY : float
        The sampling interval in the Y dimension of the image space.
    dZ : float
        The sampling interval in the Z dimension of the image space.
    XX : (nX, nY, nZ) numpy ndarray 
        The X coordinate arrays
    YY : (nX, nY, nZ) numpy ndarray 
        The Y coordinate arrays
    ZZ : (nX, nY, nZ) numpy ndarray 
        The Z coordinate arrays
    """    
    def __init__(self,nPixels=512,dPixel=1.0):
        """
        Parameters
        ----------
        nPixels : int or (3) array_like
            The number of samples in the X, Y, and Z dimensions. Default is 512
        dPixels : float or (3) array_like
            The sampling interval in the X, Y, and Z dimensions. Default is 1.0           
        """
        #Converts and assigns paramters to approprate data types
        nPixels = np.array(nPixels,dtype=int)
        if nPixels.size == 1:
            nPixels = np.repeat(nPixels,3)
        self.nX,self.nY,self.nZ = nPixels
                 
        dPixel = np.array(dPixel,dtype=float)
        if dPixel.size == 1:
            dPixel = np.repeat(dPixel,3)
        self.dX,self.dY,self.dZ = dPixel
        
        self.X = vir.censpace(self.nX,d=self.dX)
        self.Y = vir.censpace(self.nY,d=self.dY)
        self.Z = vir.censpace(self.nZ,d=self.dZ)
                                                                           
        self.XX, self.YY, self.ZZ = np.meshgrid(self.X,self.Y,self.Z)

        self.phantom = np.zeros([ self.nX, self.nY, self.nZ])

    def updatePhantom(self,phantom):
        self.phantom = phantom
        
    def updatePhantomDiscrete(self,spheres):
        self.phantom[:,:,:] = 0.0
        
        dPix = np.array((self.dX,self.dY,self.dZ))
        for S in spheres:
            self.phantom += discrete_sphere(nPixels=(self.nX, self.nY, self.nZ), \
                            center=np.array(S[:3])/dPix, radius=S[3]/dPix[0])*S[4]


def DerenzoPhantomSpheres(value=1.0,z=10.0, scale = 1.0):
    """
    Calculates the sphere paramters for a 3d Derenzo phantom.
    
    Parameters
    ----------
    value : float
        The value of the spheres. Defaults to 1.0
    z : float
        The distance to span in the Z direction. This paramtere will determine
        the maximium number of spheres that will be generated. All sphere's
        centers that  fit within the z span will be generated. Default is 10.0       
    scale : float
        Scales the size and spacing of the spheres. The current diameters of
        the spheres are [0.8, 1.0, 1.25, 1.5, 2.0, 2.5] with spacing 2x the
        diamater. Default = 1.0

    Returns
    -------
    S : (nS, 5) numpy array
        The [x,y,z,radius,value] of each spehere in the phantom
    """
    #Reads in the Derenzo csv file
    mod_dir = os.path.dirname(os.path.realpath(__file__))
    phantom_file = np.loadtxt(os.path.join(mod_dir,"Data/Phantoms/","derenzo3d.csv"),delimiter=',')

    #Seperates the spheres in two planes based on hexagonal close packing
    S_P1 = phantom_file[np.where(phantom_file[:,2] == 0)]
    S_P2 = phantom_file[np.where(phantom_file[:,2] != 0)]

    #Counts the number differnt spheres in each radii set
    radii, nS_P1 = np.unique(S_P1[:,3], return_counts=True)
    radii, nS_P2 = np.unique(S_P2[:,3], return_counts=True)

    #Delta Zs between the radii sets
    dZ = 2.*np.unique(phantom_file[:,2])[1:]

    #Number of z planes for radii set    
    nZ1 = (2*((z/2) // dZ) + 1).astype(int)
    nZ2 = (2*(((z/2 - dZ/2) // dZ) + 1)).astype(int)

    #Creates the list of sphere parameters
    S = []
    for i, r in enumerate(radii):    
        Z1 = vir.censpace(nZ1[i],d=dZ[i])
        Z2 = vir.censpace(nZ2[i],d=dZ[i])
        
        S1s = np.tile(S_P1[np.where(S_P1[:,3] == r)], (nZ1[i],1))
        S2s = np.tile(S_P2[np.where(S_P2[:,3] == r)], (nZ2[i],1))
        S1s[:,2] = np.repeat(Z1,nS_P1[i])
        S2s[:,2] = np.repeat(Z2,nS_P2[i])

        S.append(np.vstack([S1s,S2s]))

    #Create np array and scales the parameters
    S = np.vstack(S)*scale

    #Creates the vlaues column
    v = value*np.ones((S.shape[0],1))

    return np.hstack((S,v))



def discrete_sphere(nPixels=512, center=None, radius=10):
    """
        Returns a boolean array with a cicular mask with the specifiec paramters
        
        Parameters:
        
        center:    The location of the center of the mask (j,i)  [index]
        radius:    The size of the mask's radius in pixels
        img_shape: List or tuple specifying pixels [nrows,ncols] for the image
       
        Returns:
        
        A boolean array with a cicular mask
    """

    nPixels = np.array(nPixels,dtype=int)
    if nPixels.size == 1:
        nPixels = np.repeat(nPixels,3)

    if center is None:
        center = nPixels/2.0
    else:
        center = np.array(center) + nPixels/2.0
            
    x,y,z = np.indices((nPixels[0],nPixels[1],nPixels[2])) + 0.5
    
    return (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 < radius**2


def discrete_circle(nPixels=512, center=None, radius=10, upsample=1):
    """
        Returns a boolean array with a cicular mask with the specifiec paramters
        
        Parameters:
        
        center:    The location of the center of the mask (j,i)  [index]
        radius:    The size of the mask's radius in pixels
        img_shape: List or tuple specifying pixels [nrows,ncols] for the image
       
        Returns:
        
        A boolean array with a cicular mask
    """

    upsample = int(upsample)

    nPixels = upsample*np.array(nPixels,dtype=int)
    if nPixels.size == 1:
        nPixels = np.repeat(nPixels,2)

    if center is None:
        center = nPixels/2.0
    else:
        center = upsample*np.array(center) + nPixels/2.0
            
    x,y = np.indices((nPixels[0],nPixels[1])) + 0.5
    
    return vir.rebin( ((x - center[0])**2 + (y - center[1])**2 < (upsample*radius)**2), nPixels/upsample)










