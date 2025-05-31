#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 19:04:28 2020

@author: vargasp
"""
import numpy as np

def np_array(v,d,dtype=float):
    """
    Converts a scalar or arraylike variable into a numpy array with the
    number of dimensions equal to dim
    
    Parameters
    ----------
    v : scalar or array_like
        The value for the array
    d : int
        The number of dimensions of the array
    dtype : float, optional
        the data type of the new array

    Returns
    -------
    (d) numpy ndarray 
        The numpy array of dimension d and and value v
    """
    
    v = np.array(v,dtype=dtype)
    if v.size == 1:
        v = np.repeat(v,int(d))
    
    return v
                 

def censpace(n,d=1.0,c=0.0,dtype=float):
    """
    Return evenly spaced numbers over a specified interval of d centered at c
    (returned values are sample center locations)
    
    Parameters
    ----------
    n : int
        Number of samples
    d : float, optional
        Size of spacing between samples
    c : float, optional
        center of the array
    endpoint : bool, optional
        If True the coverage is the last sample. Otherwise, it is not
        included. Default is False

    Returns
    -------
    (n) numpy ndarray 
        The evenly spaced array
    """
    return np.linspace(-n+1,n-1,n,dtype=dtype)*d/2.0 + c


def boundspace(n,d=1.0,c=0.0,dtype=float):
    """
    Return evenly spaced numbers over a specified interval of d centered at c
    (returned values are sample boundary locations)
    
    Parameters
    ----------
    n : int
        Number of samples
    d : float, optional
        Size of spacing between samples
    c : float, optional
        center of the array

    Returns
    -------
    (n) numpy ndarray 
        The evenly spaced array
    """
    return np.linspace(-n,n,n+1,dtype=dtype)*d/2.0 + c


def cart2circ(x, y):
    r = np.hypot(x, y)
    az = np.arctan2(y, x)
    
    return r, az


def circ2cart(r, az):
    x = r * np.cos(az)
    y = r * np.sin(az)
    
    return x, y


def sph2cart(r, az, el):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    
    return x, y, z


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    
    return r, az, el


def sample_circle(radius=1.0, samples=128):
    """
    Calculates evenly spaced samples points on a circle.

    Parameters
    ----------
    radius : float, optional
        The radius of the sampled circle. The default is 1.0.
    samples : int, optional
        The number of sampled points on the circle. The default is 128.

    Returns
    -------
    x : (samples) numpy ndarray 
        The x coordinates of the sampled point.
    y : (samples) numpy ndarray 
        The x coordinates of the sampled point.

    """
    
    #Generates the evenly spaced angles to sample
    theta = np.linspace(0,2*np.pi,samples, endpoint=False)
    
    #Calculates the x and y coordinates
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    
    return x,y


def sample_sphere(radius=1.0,samples=512):
    """
    Calculates approximetly evenly spaced sampled points on the surface of a
    sphere using the Fibonacci Lattice method. 

    Parameters
    ----------
    radius : float, optional
        The radius of the sampled sphere.. The default is 1.0.
    samples : int, optional
        The number of sampled points on the sphere. The default is 512.

    Returns
    -------
    x : (samples) numpy ndarray 
        The x coordinates of the sampled point.
    y : (samples) numpy ndarray 
        The y coordinates of the sampled point.
    z : (samples) numpy ndarray 
        The z coordinates of the sampled point.

    """

    #Golden angle increments
    theta = np.pi*(np.sqrt(5.0) - 1.0)*np.arange(samples) 
   
    #Generates the evenly spaced angles to samples in z   
    z = np.linspace(-radius, radius, samples)
    r = np.sqrt(radius**2 - z**2)  # radius at z
    
    #Calculates the x and y coordinates
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    return x,y,z



def rebin(a, shape):
    """
    Returns the resized array of the specified dimensions.
    
    Parameters
    ----------
    a : (nX, nY) numpy ndarray 
        The array to be resampled
    shape : int or (2) array_like
        The new dimensions of new array in the X and Y dimensions. 

    Returns
    -------
    (shape[0], shape[1]) numpy ndarray 
        The evenly spaced array
    """    

    """    
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)
    """

    factors = (np.array(a.shape)/np.asanyarray(shape)).astype(int)
    sh = np.column_stack([a.shape//factors, factors]).ravel()
    return a.reshape(sh).mean(tuple(range(1, 2*a.ndim, 2)))



class Grid3d:
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
    def __init__(self,nPixels=512,dPixels=1.0,origin=0.0,dtype=float):
        """
        Parameters
        ----------
        nPixels : int or (3) array_like
            The number of samples in the X, Y, and Z dimensions. Default is 512
        dPixels : float or (3) array_like
            The sampling interval in the X, Y, and Z dimensions. Default is 1.0           
        origin : float or (3) array_like
            The center of the grid in the X, Y, and Z dimensions. Default is 0.0           
        """
        
        
        self.nPixels = np_array(nPixels,3,dtype=int)
        self.nX,self.nY,self.nZ = self.nPixels
        
        self.dPixels = np_array(dPixels,3,dtype=dtype)
        self.dX,self.dY,self.dZ = self.dPixels

        self.origin = np_array(origin,3,dtype=dtype)
        self.oX,self.oY,self.oZ = self.origin
                 
        #Creates the grid of voxel locations
        self.X = censpace(self.nX,d=self.dX,c=self.oX,dtype=dtype)
        self.Y = censpace(self.nY,d=self.dY,c=self.oY,dtype=dtype)
        self.Z = censpace(self.nZ,d=self.dZ,c=self.oZ,dtype=dtype)
        
        #Creates the grid of intersecting lines
        self.Xb = boundspace(self.nX,d=self.dX,c=self.oX,dtype=dtype)
        self.Yb = boundspace(self.nY,d=self.dY,c=self.oY,dtype=dtype)
        self.Zb = boundspace(self.nZ,d=self.dZ,c=self.oZ,dtype=dtype)
        
        #self.XX, self.YY, self.ZZ = np.meshgrid(self.X,self.Y,self.Z)


class Detector1d:
    """
    Attributes
    ----------
    nW : int
        The number of detectors along the detector's width.
    nH : int
        The number of detectors along the detector's height.
    dW : float
        The width of a dector in the width dimension.
    dH : float
        The width of a dector in the height dimension.
    sW : float
        The offset shift in the width dimension.
    sH : float
        The offset shift in the height dimension.
    XX : (nX, nY, nZ) numpy ndarray 
        The X coordinate arrays
    YY : (nX, nY, nZ) numpy ndarray 
        The Y coordinate arrays
    ZZ : (nX, nY, nZ) numpy ndarray 
        The Z coordinate arrays
    """  
    def __init__(self,nDets=512,dDet=1.0,src_iso=None,offset=0.0,det_lets=1):
        """
        Parameters
        ----------
        nDets : int or (2) array_like
            The number of detectors (samples) in the width and height
            dimensions. Default is 512
        dDets : float or (3) array_like
            The size of detectors (sampling interval) in the width and
            height dimensions. Default is 1.0
        src_iso : float
            The distance from the isocenter to the detector. Required for fan-
            beam or cone-beam geometies. Default is None
        offset : float or (2) arraylike
            The fractional offset of the detector's center and the
            project ray from the source throught the isocenter. Values
            must be within 0.0 and 1.0. The first value is width offset,
            the second is the height offset. Default is 0.0.
         det_lets : int or (2) arraylike
            The number of divisions in width and height to sub-sample a
            detector element. Default is 1.
        """
        self.nDets = int(nDets)
        self.dDet = float(dDet)
        self.sDet= offset*self.dDet
        self.src_iso = src_iso

        self.Dets = censpace(self.nDets,d=self.dDet,c=self.sDet)

        #Detectorlet Calculations
        self.nDet_lets = int(det_lets)
        self.dDet_let = self.dDet/self.nDet_lets
        self.Det_lets = censpace(self.nDets*self.nDet_lets,d=self.dDet_let,c=self.sDet)
        self.Det2_lets = np.reshape(self.Det_lets,(self.nDets,self.nDet_lets))


class Detector2d:
    """
    Attributes
    ----------
    nW : int
        The number of detectors along the detector's width.
    nH : int
        The number of detectors along the detector's height.
    dW : float
        The width of a dector in the width dimension.
    dH : float
        The width of a dector in the height dimension.
    sW : float
        The offset shift in the width dimension.
    sH : float
        The offset shift in the height dimension.
    XX : (nX, nY, nZ) numpy ndarray 
        The X coordinate arrays
    YY : (nX, nY, nZ) numpy ndarray 
        The Y coordinate arrays
    ZZ : (nX, nY, nZ) numpy ndarray 
        The Z coordinate arrays
    """  
    def __init__(self,nDets=512,dDet=1.0,src_iso=None,offset=0.0,det_lets=1):
        """
        Parameters
        ----------
        nDets : int or (2) array_like
            The number of detectors (samples) in the width and height
            dimensions. Default is 512
        dDets : float or (3) array_like
            The size of detectors (sampling interval) in the width and
            height dimensions. Default is 1.0
        src_iso : float
            The distance from the isocenter to the detector. Required for fan-
            beam or cone-beam geometies. Default is None
        offset : float or (2) arraylike
            The fractional offset of the detector's center and the
            project ray from the source throught the isocenter. Values
            must be within 0.0 and 1.0. The first value is width offset,
            the second is the height offset. Default is 0.0.
         det_lets : int or (2) arraylike
            The number of divisions in width and height to sub-sample a
            detector element. Default is 1.
        """
        nDets = np.array(nDets,dtype=int)
        if nDets.size == 1:
            nDets = np.repeat(nDets,2)
        self.nW,self.nH = nDets
                 
        dDet = np.array(dDet,dtype=float)
        if dDet.size == 1:
            dDet = np.repeat(dDet,2)
        self.dW,self.dH = dDet
        
        offset = np.array(offset,dtype=float)
        if offset.size == 1:
            offset = np.append(offset, 0.0)
        self.sW,self.sH = offset*dDet

        self.src_iso = src_iso

        self.W = censpace(self.nW,d=self.dW,c=self.sW)
        self.H = censpace(self.nH,d=self.dH,c=self.sH)

        #Detectorlet Calculations
        det_lets = np.array(det_lets,dtype=int)
        if det_lets.size == 1:
            det_lets = np.repeat(det_lets,2)
        self.nW_lets,self.nH_lets = det_lets

        self.dW_let = self.dW/self.nW_lets
        self.dH_let = self.dH/self.nH_lets
        self.W_lets = censpace(self.nW*self.nW_lets,d=self.dW_let,c=self.sW)
        self.H_lets = censpace(self.nH*self.nH_lets,d=self.dH_let,c=self.sH)
        self.W2_lets = np.reshape(self.W_lets,(self.nW,self.nW_lets))
        self.H2_lets = np.reshape(self.H_lets,(self.nH,self.nH_lets))

        #Not clear --NEEDS UPDATING--
        self.X = self.W
        self.Y = 0.0
        self.Z = self.H

        self.XX, self.YY, self.ZZ = np.meshgrid(self.X,self.Y,self.Z)


    def center(self):
        return (self.X, self.Y, self.Z)

    def pairs(self):
        pair1 = []
        pair2 = []

        for z in self.Z:
            pair1.append((self.X[0], self.Y, z))
            pair2.append((self.X[-1], self.Y, z))
                
        return (pair1,pair2)


class Source2d:
    def __init__(self,center=(0,-1024,0)):
        self.X = center[0]
        self.Y = center[1]
        self.Z = center[2]

    def center(self):
        return (self.X, self.Y, self.Z)


class Geom:
    """
    Attributes
    ----------
    nViews : int
        The number of projection angles
    Views : np.array
        The array of projections angles
    dView : float
        The angle between projection angles in radians.        
    coverage : float
        The angular coverage in radians.
    angle0 : float, optional
        The intial projection angle in radians.
    pitch : float
        The z translation of the phantom per 2*pi rotation.
    nRotations : float
        The number of rotations 
    nAngles : int
        The number of projection angles in one 2*pi rotation.
    dZ : float
        The z distance translated between projection angles in units of the
        detector height
    Z : np.array
        The Z coordinates of the source with respect to the center of the
        phantom in units of detector height
    
    Methods
    -------
    updateViews(self, Views)
        Updates the array of projection angles with a custom array 
    """
    
    def __init__(self,nViews=512, coverage=2.0*np.pi, angle0=0.0, pitch=0.0, \
                 endpoint=False, src_iso=np.inf, src_det=np.inf,fan_angle=0.0,
                 zTran=0.0):
        """
        Parameters
        ----------
        nViews : int, optional
            The number of projection angles in the sinogram. Default is 512
        coverage : float, optional
            The angular coverage in radians (Multiple revolutions can be used
            for helical tragectories). Default is 2*pi
        angle0 : float, optional
            The intial projection angle in radians. Default is 0
        pitch : float, optional
            The z translation of the phantom per 2*pi rotation. Default is 0
        endpoint : bool, optional
            If True the coverage is the last sample. Otherwise, it is not
            included. Default is False
        src_iso : float, optional
            Distance from the source to the isocenter. Required for fanbeam.
            Default is np.inf
        src_det : float, optional
            Distance from the source to the detector. Required for fanbeam.
            Default is np.inf            
        """
        
        #Reads in arguments
        self.nViews = int(nViews) #[Unitless]
        self.coverage = coverage #[Radians]
        self.angle0 = angle0 #[Radians]
        self.pitch = pitch #[Number of detector arrays / rotations]
        self.endpoint = endpoint #Flag
        self.src_iso = src_iso #[]
        self.src_det = src_det
        self.fan_angle = fan_angle #[Radians]
        self.zTran = zTran #[Number of detector rows]

        self.Views, self.dView = np.linspace(self.angle0, \
                self.angle0+self.coverage,self.nViews, \
                endpoint=self.endpoint,retstep=True)
        self.nRotations = coverage / (2.0*np.pi)

        nAngles = 2*np.pi/self.dView
        if np.isclose(nAngles,round(nAngles)):
            self.nAngles = round(nAngles)
        else:
            self.nAngles =nAngles
            #raise ValueError("dView must equally divide 2pi ")  
            print("Warning! dView doesn't equally divide 2pi ")
            
        #Geometry specific parameters
        if self.fan_angle != 0.0:
            if self.src_iso == np.inf:
                raise ValueError("Fanbeam set, src_iso must be provided")     
            if self.src_det == np.inf:
                raise ValueError("Fanbeam set, src_det must be provided")     
                
            self.geom = "Fan beam"
            self.fov = 2.0*self.src_iso*np.sin(self.fan_angle/2.0)
        else:
            self.geom = "Parallel beam"
        
        #Trajectory Parameters
        self.dZ = self.zTran * self.nRotations / self.nViews 
        self.Z = censpace(self.nViews, self.dZ)
        

    def updateViews(self,Views,dView=None,coverage=None,Z=None):
        """
        Updates the array of projection angles with a custom array 
            updateViews(self, Views)
        
        Parameters
        ----------
        Views : array_like
            An array of projection angles in radians
        dView : float, optional
            The angle between projection angles in radians. Default is None
        pitch : float
            The z translation of the phantom per 2*pi rotation. Default is None
        coverage : float, optional
            The angular coverage in radians. Default is None
        """
        
        self.Views = np.array(Views)
        
        if dView is None:
            self.dView = np.mean(Views[1:] - Views[:-1])

        if coverage is None:
            self.coverage = self.Views[-1] - self.Views[0] + self.dView
        
        self.nViews = self.Views.size
        self.angle0 = self.Views[0]
        

        self.nRotations = self.coverage / (2.0*np.pi)
        if Z is None:
            self.Z = censpace(self.nViews, self.dZ)
        else:
            self.Z = Z

    def interpZ(self,intZ,angle,nRows,all_views=False):
        """
        Calculates the indices and distance of the closest detector rows and 
        their distancs at a specific aquiared angle to slice(s) of interest in
        z.

        Parameters
        ----------
        intZ : scalar or array_like
            The z locations of interest in a helical sinogram.
        angle : int
            The index of the angle in the trajectory.
        nRows : int
            The number of rows in the detector.
        all_views : bool, optional
            Returns all of the parameters for every rotation at the desired
            angle. The default is False.

        Returns
        -------
        idxV : int or np array of ints
            The view index(es) of the corresponding to the angle.
        idxL : int or np array of ints
            The index(es) of the closest row(s) below the slice of interest.
        idxU : int or np array of ints
            The index(es) of the closest row(s) above the slice of interest.
        dL : float or np array of float
            The distance of the closest row(s) below to the slice of interest.
        dU : float or np array of float
            The distance of the closest row(s) above to the slice of interest.
        """
        intZ = np.array(intZ)

        #Ensures the angle is [0, 2*pi)
        angle = angle % (2.0*np.pi)

        #Calculates all projection views at angle 
        angles = np.arange(np.ceil(self.nRotations)+1)*2.0*np.pi + angle
        #angles = angles[(angles < self.Views[-1]) | np.isclose(angles, self.Views[-1])]

        #Calculates the indices of the views that equal or the closest below
        #the projection view
        idxVp = np.searchsorted(self.Views, angles, side='left')
        idxVp = np.where(np.isclose(self.Views[idxVp-1], angles), idxVp-1, idxVp)
        idxVp = idxVp % self.nViews

        #Calculates the indices of the views that equal or the closest above
        #the projection view
        idxVn = np.where(np.isclose(self.Views[idxVp], angles), idxVp, idxVp-1)
        idxVn = idxVn % self.nViews    

        #Removes duplicate pairs
        if idxVn[-1] == idxVn[-2] and idxVp[-1] == idxVp[-2]:
            idxVn = idxVn[:-1]
            idxVp = idxVp[:-1]
            angles = angles[:-1]

        #Acquired Zs at each projection view
        acqZ_Vp = np.add.outer(self.Z[idxVp], censpace(nRows))
        acqZ_Vn = np.add.outer(self.Z[idxVn], censpace(nRows))

        #Indices of z slices directly below and above interpolated slice
        idxRn_Vp = np.zeros((idxVp.size,intZ.size), dtype=int)
        idxRp_Vp = np.zeros((idxVp.size,intZ.size), dtype=int)
        
        idxRn_Vn = np.zeros((idxVn.size,intZ.size), dtype=int)
        idxRp_Vn = np.zeros((idxVn.size,intZ.size), dtype=int)
     
        #Distances the acquired slice is below and above the interpolated slice
        dRn_Vp = np.zeros((idxVp.size,intZ.size), dtype=float)
        dRp_Vp = np.zeros((idxVp.size,intZ.size), dtype=float)
        
        dRn_Vn = np.zeros((idxVn.size,intZ.size), dtype=float)
        dRp_Vn = np.zeros((idxVn.size,intZ.size), dtype=float)

        #Distances between angles 
        dVn = np.tile(self.Views[idxVn] - angles,[intZ.size,1]).T
        dVp = np.tile(self.Views[idxVp] - angles,[intZ.size,1]).T

        #dVp = np.where(dVp < 0, dVp % (2*np.pi), dVp)
        dVp = dVp % (2*np.pi)
        
        dVn = np.where(np.isclose(dVn,0.0), 0.0, dVn)
        dVp = np.where(np.isclose(dVp,0.0), 0.0, dVp)
        dVn = np.where(np.isclose(dVn,2*np.pi), 0.0, dVn)        
        dVp = np.where(np.isclose(dVp,2*np.pi), 0.0, dVp)
        
        
        #Loops through all of the projection angles and finds the nearest slice
        #below, above, and the distances to the interpolated slice
        for i in range(angles.size):

            idxRp = np.array(np.searchsorted(acqZ_Vp[i,:],intZ))
            idxRn = np.where(np.isclose(acqZ_Vp[i,idxRp.clip(0,nRows-1)],intZ), idxRp, idxRp-1)
            idxRp_Vp[i,:] = idxRp.clip(0,nRows-1)
            idxRn_Vp[i,:] = idxRn.clip(0,nRows-1)


            idxRp = np.array(np.searchsorted(acqZ_Vn[i,:],intZ))
            idxRn = np.where(np.isclose(acqZ_Vn[i,idxRp.clip(0,nRows-1)],intZ), idxRp, idxRp-1)
            idxRp_Vn[i,:] = idxRp.clip(0,nRows-1)            
            idxRn_Vn[i,:] = idxRn.clip(0,nRows-1)


            dRn_Vp[i,:] = acqZ_Vp[i,idxRn_Vp[i,:]] - intZ
            dRp_Vp[i,:] = acqZ_Vp[i,idxRp_Vp[i,:]] - intZ

            dRn_Vn[i,:] = acqZ_Vn[i,idxRn_Vn[i,:]] - intZ
            dRp_Vn[i,:] = acqZ_Vn[i,idxRp_Vn[i,:]] - intZ

            
            
            
        """
        print(idxVn)
        print(acqZ_Vn)
        print(idxRn_Vn)
        print(dRn_Vn)

        print(idxRp_Vn)
        print(dRp_Vn)
        """

        """
        print()
        print(f"Ang: {angle:.3f}, All angles: {angles}")
        print(f"L Views: {np.around(self.Views[idxVn],3)}, idx: {idxVn}, \ndV:\n{np.around(dVn,3)}")
        print(f"R Views: {np.around(self.Views[idxVp],3)}, idx: {idxVp}, \ndV:\n{np.around(dVp,3)}")
        print(f"L Row:\n{np.around(dRn_Vp,3).T}\nidx:\n{idxRn_Vp.T}")
        """
        
        #Finds the first projection angle with the slices closest to the 
        #interpolated slice
        if  all_views:
            idxR_Vp = np.stack([idxRn_Vp,idxRp_Vp],axis=-1)
            dR_Vp = np.stack([dRn_Vp,dRp_Vp],axis=-1)
            idxR_Vn = np.stack([idxRn_Vn,idxRp_Vn],axis=-1)
            dR_Vn = np.stack([dRn_Vn,dRp_Vn],axis=-1)
            
            return idxVp, dVp, idxR_Vp, dR_Vp, idxVn, dVn, idxR_Vn, dR_Vn
        
        else:
            idxZ = range(intZ.size)
            
            dRn_Vp = np.where(dRn_Vp<=0,dRn_Vp,-np.inf)
            dRp_Vp = np.where(dRp_Vp>=0,dRp_Vp,np.inf)
            dRn_Vn = np.where(dRn_Vn<=0,dRn_Vn,-np.inf)
            dRp_Vn = np.where(dRp_Vn>=0,dRp_Vn,np.inf)

            #Find the indices of the closest rows below and above z
            idxRn = np.sqrt(dRn_Vp**2 + (dVp/self.dView)**2).argmin(axis=0)
            idxRp = np.sqrt(dRp_Vp**2 + (dVp/self.dView)**2).argmin(axis=0)

            

            #Indices and distances of views above the nearest view
            #above the desired view
            idxVp = np.stack([idxVp[idxRn],idxVp[idxRp]],axis=-1)
            dVp = np.stack([dVp[idxRn,idxZ],dVp[idxRp,idxZ]],axis=-1)
            
            #Indices and distances of rows closest to z at the nearest view
            #above the desired view
            idxR_Vp = np.stack([idxRn_Vp[idxRn,idxZ],idxRp_Vp[idxRp,idxZ]],axis=-1)
            dR_Vp = np.stack([dRn_Vp[idxRn,idxZ],dRp_Vp[idxRp,idxZ]],axis=-1)

            #Find the indices of the closest rows below and above z
            idxRn = np.sqrt(dRn_Vn**2 + (dVn/self.dView)**2).argmin(axis=0)
            idxRp = np.sqrt(dRp_Vn**2 + (dVn/self.dView)**2).argmin(axis=0)
            
            #Indices and distances of views above the nearest view
            #below the desired view
            idxVn = np.stack([idxVn[idxRn],idxVn[idxRp]],axis=-1)
            dVn = np.stack([dVn[idxRn,idxZ],dVn[idxRp,idxZ]],axis=-1)

            #Indices and distances of rows closest to z at the nearest view
            #below the desired view
            idxR_Vn = np.stack([idxRn_Vn[idxRn,idxZ],idxRp_Vn[idxRp,idxZ]],axis=-1)
            dR_Vn = np.stack([dRn_Vn[idxRn,idxZ],dRp_Vn[idxRp,idxZ]],axis=-1)
 
    
            """
            print("\nClosest")
            print(f"L Views:\n{np.around(self.Views[idxVn],3).T}\nidx:\n{idxVn.T} \
                  \ndV:\n {np.around(dVn,3).T}")
            print(f"L Row:\n{np.around(dR_Vn,3).T}\nidx:\n{idxR_Vn.T}")
            print(f"R Views:\n{np.around(self.Views[idxVp],3).T}\nidx:\n{idxVp.T} \
                  \ndV:\n {np.around(dVp,3).T}")
            print(f"R Row:\n{np.around(dR_Vp,3).T}\nidx:\n{idxR_Vp.T}")
            """
            return idxVp, dVp, idxR_Vp, dR_Vp, idxVn, dVn, idxR_Vn, dR_Vn


    
    def bins(self,nBins):
        dBin = self.fan_angle/nBins*self.src_det
        return censpace(nBins, dBin)


    def gammas(self,nGammas):
        dGamma = self.fan_angle/nGammas
        return censpace(nGammas, dGamma)

        
def mask(image, radius=None, fill=0.0):
    """
        Returns a boolean array with a cicular mask with the specifiec paramters
        
        Parameters:
        
        center:    The location of the center of the mask (j,i)  [index]
        radius:    The size of the mask's radius in pixels
        img_shape: List or tuple specifying pixels [nrows,ncols] for the image
       
        Returns:
        
        A boolean array with a cicular mask
    """

    nX, nY = image.shape    
    x,y = np.indices((nX,nY))
    
    if radius is None:
        radius = min(nX,nY)/2.0
    
    return image*((x - nX/2.0)**2 + (y - nY/2.0)**2 <= radius**2)
    
    
    
    
    """
    
    
    if circle:
        shape_min = min(image.shape)
        radius = shape_min // 2
        img_shape = np.array(image.shape)
        coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]])
        dist = ((coords - img_shape // 2) ** 2).sum(0)
        outside_reconstruction_circle = dist > radius ** 2
        if np.any(image[outside_reconstruction_circle]):
            warn('Radon transform: image must be zero outside the '
                 'reconstruction circle')
        # Crop image to make it square
        slices = tuple(slice(int(np.ceil(excess / 2)),
                             int(np.ceil(excess / 2) + shape_min))
                       if excess > 0 else slice(None)
                       for excess in (img_shape - shape_min))
        padded_image = image[slices]

    """