#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 15:41:27 2021

@author: vargasp
"""

import numpy as np
import vir
import ctypes

import vir.mpct as mpct


try:
    c_test = ctypes.CDLL(__file__.rsplit("/",1)[0] + "/siddons.so")
except:
    print("Warning. PSiddons shared object libray not present.")




def list_ctypes_object(sdlist, flat=True):
    """
    Converts the data in unraveled array created by siddons to C data types.
    WARNING!!! It's not clear if this process creates a copy of the data or
    pointers to the data. 

    Parameters
    ----------
    sdlist : (...) numpy ndarray of objects
        the returned array is of shape (...) with 2 lists of the raveled pixel
        index and intersection lengths or 4 listst of the unraveled pixel 
        indexes and intersection lengths from siddons 

    Returns
    -------
    sdlist_c : (...) numpy ndarray of C pointers 
        the returned array is of shape (...) with 2 lists of the raveled pixel
        index and intersection lengths or 4 listst of the unraveled pixel 
        indexes and intersection lengths from siddons       
    """
       
    if flat:
        sdlist_c = (mpct.Ray*np.prod(sdlist.shape))()
        count = list_count_elem(sdlist).flatten()
        for ray_idx, ray in enumerate(sdlist.flatten()):
            sdlist_c[ray_idx].n = ctypes.c_int(count[ray_idx])
 
            if ray != None:
                sdlist_c[ray_idx].X = np.ctypeslib.as_ctypes(ray[0].astype(np.int32))
                sdlist_c[ray_idx].Y = np.ctypeslib.as_ctypes(ray[1].astype(np.int32))
                sdlist_c[ray_idx].Z = np.ctypeslib.as_ctypes(ray[2].astype(np.int32))
                sdlist_c[ray_idx].L = np.ctypeslib.as_ctypes(ray[3].astype(np.float32))
        
        sdlist_c = ctypes.byref(sdlist_c)
        
    else:
        sdlist_c = np.empty(sdlist.shape, dtype=np.object)
        count = list_count_elem(sdlist)

        for ray_idx, ray in np.ndenumerate(sdlist):
            if ray != None:
                sdlist_c[ray_idx] = mpct.ctypes_ray(count[ray_idx],ray[0].astype(np.int32),\
                                               ray[1].astype(np.int32),ray[2].astype(np.int32), \
                                               ray[3].astype(np.float32))
                

    return sdlist_c




def list_del_ind(sdlist, X_ind, Y_ind, Z_ind, ravel=False):
    """
    Remomves sdlist ray elements that are not included in the argument
    paramters. Reduces the size efficeint processing of local areas
    
    Parameters
    ----------
    sdlist : (...) numpy ndarray of objects
        the returned array is of shape (...) with 2 lists of the raveled pixel
        index and intersection lengths or 4 listst of the unraveled pixel 
        indexes and intersection lengths from siddons 
    X_ind :  array_like
        The integer values to include in the trimmed list 
    Y_ind :  array_like
        The integer values to include in the trimmed list 
    Z_ind :  array_like
        The integer values to include in the trimmed list 
    Ravel : bool
        Flag indicating if the sdlist is raveled or unraveled. Default is ravel

    Returns
    -------
    (...) numpy ndarray of objects
        the returned array is of shape (...) with 2 lists of the unraveled pixel
        indices and intersection lengths. The intersection indices of sdlist are
        shifted 
    """
    
    if ravel == True:
        raise ValueError('Cuurently unsuported, ravel must be False')
    
    #Iterates through all of the lists    
    for ray_idx, ray in np.ndenumerate(sdlist):
        if ray != None:
            X = np.isin(ray[0], X_ind)
            Y = np.isin(ray[1], Y_ind)
            Z = np.isin(ray[2], Z_ind)

            idx = np.logical_and(X, np.logical_and(Y, Z))
            
            sdlist[ray_idx] = [ray[0][idx],ray[1][idx],ray[2][idx],ray[3][idx]]

    return sdlist


def list_index_shift(sdlist, shift, ravel=False):
    """
    Shifts the index v
    
    Parameters
    ----------
    sdlist : (...) numpy ndarray of objects
        the returned array is of shape (...) with 2 lists of the raveled pixel
        index and intersection lengths or 4 listst of the unraveled pixel 
        indexes and intersection lengths from siddons 
    shift : (3) array_like
        The integer vlaue to shift in the X, Y, and Z dimensions 
    Ravel : bool
        Flag indicating if the sdlist is raveled or unraveled. Default is ravel

    Returns
    -------
    (...) numpy ndarray of objects
        the returned array is of shape (...) with 2 lists of the unraveled pixel
        indices and intersection lengths. The intersection indices of sdlist are
        shifted 
    """
    
    
    shift = np.array(shift, dtype=int)
    
    if ravel == True:
        raise ValueError('Cuurently unsuported, ravel must be False')
    
    #Iterates through all of the lists    
    for ray_idx, ray in np.ndenumerate(sdlist):
        if ray != None:
            sdlist[ray_idx] = ray_index_shift(ray, shift)

    return sdlist


def list2array_iter_ray(ray, array, ravel=False):
    """
    Sums a single ray of a siddons array of lists to a 3d array
    
    Parameters
    ----------
    ray : (4) or (2) (nElem) numpy arrays
        A list containing 4 or 2 np.arrays of voxel indices and intersection
        lengths all with size of nElems         
    array : (X,Y,Z) numpy array
        a 3d array of summations intersection lengths 
    ravel : bool
        Flag indicating if the sdlist is raveled or unraveled. Default is ravel

    Returns
    -------
    [(X,Y,Z) numpy array
        the returned 3d array of summations interscetion of a ray
    """
    #Loops through the flat array incrementing the intersection length
    #The loop is required due to python indexing will NOT increment
    #duplicate indices
    for i in range(ray[0].size):
        if ravel:
            array.flat[ray[0][i]] += ray[1][i]
        else:
            array[ray[0][i],ray[1][i],ray[2][i]] += ray[3][i]
    
    

def list2array(sdlist, nPixels, ravel=False, flat=False):
    """
    Converts a siddons array of lists into a 3d array of the summation of the
    intersctions lengths associated with each voxel
    
    Parameters
    ----------
    sdlist : (...) numpy ndarray of objects
        the returned array is of shape (...) with 2 lists of the raveled pixel
        index and intersection lengths or 4 listst of the unraveled pixel 
        indexes and intersection lengths from siddons 
    nPixels : (3) array_like
        The number of voxels in the X, Y, and Z dimensions
    ravel : bool, optional
        Stores the pixel values as either a raveled array index array (a
        single integer value representing all three dimesions) or an unraveled
        array index (3 arrays corresponding to the X, Y, and Z indices)
        dimensional array. Default is unravled
        
    Returns
    -------
    [(X,Y,Z) numpy array
        the returned 3d array of summations intersection lengts of a sdlist
    """
    
    #Defines the intersction lenght array
    iter_array = np.zeros(nPixels, dtype=np.float32)
    
    
    if flat == True:
        list2array_iter_ray(sdlist, iter_array, ravel=True)
            
    else:
        #Loops through all rays in the list
        for ray_idx, ray in np.ndenumerate(sdlist):
            if ray != None:
                list2array_iter_ray(ray, iter_array, ravel=ravel)


    return iter_array


def list_count_elem(sdlist):
    """
    Counts the number of elements in a siddons object array
        
    Parameters
    ----------
    sdlist : (...) numpy ndarray of objects
        The returned array of siddons with a 2 or 4 lists of the raveled or 
        unraveled pixel indices with intersection lengths
 
    Returns
    -------
    Count : (...) numpy ndarray 
        The number of elements in each element of the siddons object array
    """
    
    
    if isinstance(sdlist, np.ndarray):
        count = np.zeros(sdlist.shape, dtype=int)    
    else:
        return sdlist[0].size
        
    for ray_idx, ray in np.ndenumerate(sdlist):
        if ray != None:
            count[ray_idx] = ray[0].size
            
    return count









            


def list_flip(sdlist,x=None,y=None,z=None,ravel=False,nPixels=None):

    for ray_idx, ray in np.ndenumerate(sdlist):
        if ray != None:
            sdlist[ray_idx] = ray_flip(ray,x=x,y=y,z=z,ravel=ravel,nPixels=nPixels)
            
    return sdlist


def ray_flip(ray,x=None,y=None,z=None,ravel=False,nPixels=None):
    """
    Flips saingle ray in an object array created by Siddons along an axis.

    Parameters
    ----------
    ray : (4) (nElem) numpy arrays
        The list of raveled indices
    x : int
        Axis 
    
    nPixels : (3) array_like
        The number of voxels in the X, Y, and Z dimensions

    Returns
    -------
    [(nElem) numpy array, (nElem) numpy array]
        the returned array is of shape (...) with 2 lists of the raveled
        pixel indices and intersection length
    """
    
    if ravel:
        ray = ray_unravel(ray,nPixels)
    
    if x:
        ray = [(2*x - ray[0]).astype(int), ray[1], ray[2], ray[3]]
    if y:
        ray = [ray[0], (2*y - ray[1]).astype(int), ray[2], ray[3]]
    if z:
        ray = [ray[0], ray[1], (2*z - ray[2]).astype(int), ray[3]]
    
    if ravel:    
        return [np.ravel_multi_index((ray[0],ray[1],ray[2]),nPixels),ray[3]]
    else:
        return ray


def ray_index_shift(ray, shift):   
    return [ray[0]+shift[0],ray[1]+shift[1],ray[2]+shift[2],ray[3]]

def ray_boundary(ray,nPixels):
    
    Xidx = np.logical_and(ray[0]>=0, ray[0]<nPixels[0])
    Yidx = np.logical_and(ray[1]>=0, ray[1]<nPixels[1])
    Zidx = np.logical_and(ray[2]>=0, ray[2]<nPixels[2])
    
    idx = np.logical_and(Xidx, np.logical_and(Zidx, Yidx))

    return [ray[0][idx],ray[1][idx],ray[2][idx],ray[3][idx]]


def list_convole(sdlist, kernel, ravel=False, nPixels=None, axis=None):

    if len(axis) != kernel.ndim:
        raise ValueError('Number of axis must match kernel dimensions')


    for dim in kernel.shape:
        if dim%2 == 0:
            raise ValueError('Kernel must have an odd numer of elements in each dimension')

    k_dims = kernel.shape
    sd_dims = sdlist.shape
    sdlist_convole = np.empty(sdlist.shape, dtype=object)

    for ray_idx, ray in np.ndenumerate(sdlist):
        ray_idx_c = list(ray_idx)
        ker_idx = []
        
        for i, (ax,k_dim) in enumerate(zip(axis,k_dims)):
            l = ray_idx_c[ax] - int(k_dim/2)
            if l < 0:
                lk = abs(l)
                l = 0
            else:
                lk = 0
                            
            u = ray_idx_c[ax] + int(k_dim/2) + 1
            if u > sd_dims[ax]:
                uk = k_dims[i] - (u - sd_dims[ax])
                u = sd_dims[ax]
            else:
                uk = k_dims[i]
     
            ray_idx_c[ax] = slice(l,u)
            ker_idx.append(slice(lk,uk))
        
        kF = kernel.sum()/kernel[tuple(ker_idx)].sum()
        
        sdlist_convole[ray_idx] = rays_weight_ave(sdlist[tuple(ray_idx_c)],\
                                                  kF*kernel[tuple(ker_idx)],ravel=ravel,nPixels=nPixels)

    return sdlist_convole


def list_weight_ave(sdlist, weights, ravel=False,nPixels=None, axis=None):


    if len(axis) != weights.ndim:
        raise ValueError('Number of axis must match weights dimensions')

    if axis == None:
        return rays_weight_ave(sdlist, weights, ravel=ravel, nPixels=nPixels)

    #Creates the new shape based on axis averaging and moves averaged
    #dimensions to the end
    sdlist_ave = np.empty(np.delete(sdlist.shape, axis), dtype=object)
    sdlist = np.moveaxis(sdlist, axis, np.arange(-1*(np.array(axis).size),0))
   
    #Loops through all rays in the list
    for ray_idx, ray in np.ndenumerate(sdlist_ave):
        sdlist_ave[ray_idx] = rays_weight_ave(sdlist[ray_idx], weights, ravel=ravel,\
                        nPixels=nPixels)
    
    return sdlist_ave


def rays_weight_ave(sdlist, weights, ravel=False, nPixels=None):

    if ravel == False and nPixels == None:
        raise ValueError('nPixels must be provided if ravel=False')

    #Loops through all rays in the list
    sdlist_weight = np.empty(sdlist.shape, dtype=object)
    
    for ray_idx, ray in np.ndenumerate(sdlist):
        if ray == None:
            sdlist_weight[ray_idx] = None
        else:            
            if ravel == False:
                ray = ray_ravel(ray,nPixels)
            
            sdlist_weight[ray_idx] = [ray[0],weights[ray_idx]*ray[1]]

    flat_ind, flat_len = list_flatten(sdlist_weight, nPixels=nPixels,ravel=True)
        
    #Finds all of the unique elemets and thier positions
    flat_ind_ave, idx = np.unique(flat_ind, return_inverse=True)

    #If no rays have values return None
    if flat_ind_ave.size == 0:
        return None
    
    flat_len_ave = np.zeros(flat_ind_ave.size, dtype=np.float32)
    #Creates and calculates the average length per intersection
    for i, ix in enumerate(idx):
        flat_len_ave[ix] += flat_len[i]

    #Returns the list of indices and lengths in the orginal format
    if ravel == True:
        return [flat_ind_ave, flat_len_ave]
    else:
        return ray_unravel([flat_ind_ave, flat_len_ave], nPixels)


def list_ave(sdlist, ravel=False, flat=False, nPixels=None, nRays=None, axis=None):
    """
    Combines and averages the intersection lengths and pixel indices over
    multiple rays in in object array. 
        
    Parameters
    ----------
    sdlist : (...) numpy ndarray of objects
        The returned array of siddons with a 2 or 4 lists of the raveled or 
        unraveled pixel indices with intersection lengths
    ravel : boolean
        The The coefficients of the parametric lines
    nPixels : (3) array_like
        The number of voxels in the X, Y, and Z dimensions
    axis : sclar or array_like
        The axis or axes to average over   
    
    Returns
    -------
    Pts : (3) or (...,3) array_like
        The coordinates of the intersection poins. If the line lies within the
        plane inf for the coordinates are returned. If the line lies parallel
        to, but no in the plane nan for the coordinates are returned. 
    """   
    if flat == True and axis != None:
        raise ValueError('Axis cannot be provided when flat=True')

    if axis == None:
        return rays_ave(sdlist, ravel=ravel, flat=flat,\
                        nPixels=nPixels, nRays=nRays)

    #Creates the new shape based on axis averaging and moves averaged
    #dimensions to the end
    sdlist_ave = np.empty(np.delete(sdlist.shape, axis), dtype=object)
    sdlist = np.moveaxis(sdlist, axis, np.arange(-1*(np.array(axis).size),0))
   
    #Loops through all rays in the list
    for ray_idx, ray in np.ndenumerate(sdlist_ave):
        sdlist_ave[ray_idx] = rays_ave(sdlist[ray_idx], ravel=ravel, flat=flat,\
                        nPixels=nPixels, nRays=nRays)
    
    return sdlist_ave

    
def rays_ave(sdlist, ravel=False, flat=False, nPixels=None, nRays=None):
    """
    Combines and averages the intersection lengths and pixel indices over
    multiple rays in in object array.
        
    Parameters
    ----------
    sdlist : (...) numpy ndarray of objects
        The returned array of siddons with a 2 or 4 lists of the raveled or 
        unraveled pixel indices with intersection lengths
    nPixels : (3) array_like
        The number of voxels in the X, Y, and Z dimensions
    ravel : boolean
        The The coefficients of the parametric lines
 
    Returns
    -------
    Pts : (3) or (...,3) array_like
        The coordinates of the intersection poins. If the line lies within the
        plane inf for the coordinates are returned. If the line lies parallel
        to, but no in the plane nan for the coordinates are returned. 
    """   
    if ravel == False and nPixels == None:
        raise ValueError('nPixels must be provided if ravel=False')

    if flat == True and nRays == None:
        raise ValueError('nRays must be provided if flat=True')

    if flat == False and nPixels == None:
        raise ValueError('nPixels must be provided if flat=False')
    
    #Flatten the array to get the indices and lengths arrays
    if flat == True:
        flat_ind, flat_len = sdlist
    else:
        nRays = sdlist.size
        flat_ind, flat_len = list_flatten(sdlist, nPixels=nPixels,ravel=ravel)
        
    #Finds all of the unique elemets and thier positions
    flat_ind_ave, idx = np.unique(flat_ind, return_inverse=True)

    #If no rays have values return None
    if flat_ind_ave.size == 0:
        return None
    
    flat_len_ave = np.zeros(flat_ind_ave.size, dtype=np.float32)
    #Creates and calculates the average length per intersection
    for i, ix in enumerate(idx):
        flat_len_ave[ix] += flat_len[i]
    flat_len_ave /= nRays

    #Returns the list of indices and lengths in the orginal format
    if ravel == True or flat == True:
        return [flat_ind_ave, flat_len_ave]
    else:
        return ray_unravel([flat_ind_ave, flat_len_ave], nPixels)


def list_flatten(sdlist,ravel=False,nPixels=None):
    """
    Flattens an unraveled or raveld object array created by Siddons. Stores the
    data in two np.arrays with raveled pixel indices and intersection lengths.

    Parameters
    ----------
    sdlist : (...) numpy ndarray of objects
        The returned array of siddons with a 4 arrays of the unraveled voxel
        indices and intersection lengths
    nPixels : (3) array_like
        The number of voxels in the X, Y, and Z dimensions associated with the
        object array
    ravel : bool, optional
        Stores the pixel values as either a raveled array index array (a
        single integer value representing all three dimesions) or an unraveled
        array index (3 arrays corresponding to the X, Y, and Z indices)
        dimensional array. Default is unravled

    Returns
    -------
    [(nElem) numpy array, (nElem) numpy array]
        Flattens an unraveled or raveld object array created by Siddons.
    """
    if ravel == False and nPixels == None:
        raise ValueError('nPixels must be provided if ravel=False')
    
    #Counts the cumulative total of elments and adds a 0 to the first index
    count = np.insert(np.cumsum(list_count_elem(sdlist)),0,0)
    
    #Creates the the two arrays for indices and length
    flat_ind = np.zeros(count[-1],dtype=int)
    flat_len = np.zeros(count[-1],dtype=np.float32)

    #Iterates through the list
    for idx, ray in enumerate(sdlist.flat):
        if ray != None:
            
            if ravel == False:
                ray = ray_ravel(ray,nPixels)
         
            #Copies the ray's elements to the newly created arrays
            flat_ind[count[idx]:count[idx+1]] = ray[0]
            flat_len[count[idx]:count[idx+1]] = ray[1]

    return flat_ind, flat_len


def list_ravel(sdlist,nPixels):
    """
    Ravels an unraveled object array created by Siddons. This function will
    not create copied matrix, but chages it in place.

    Parameters
    ----------
    sdlist : (...) numpy ndarray of objects
        The returned array of siddons with a 4 arrays of the unraveled voxel
        indices and intersection lengths
    nPixels : (3) array_like
        The number of voxels in the X, Y, and Z dimensions associated with the
        object array

    Returns
    -------
    [(nElem) numpy array, (nElem) numpy array]
        the returned array is of shape (...) with 2 arrays of the raveled voxle
        index and intersection lengths of the ray.
    """
    for ray_idx, ray in np.ndenumerate(sdlist):
        if ray != None:
            sdlist[ray_idx] = ray_ravel(ray,nPixels)
            
    return sdlist


def list_unravel(sdlist,nPixels):
    """
    Unravels a raveled object array created by Siddons. This function will
    not create copied matrix, but chages it in place.

    Parameters
    ----------
    sdlist : (...) numpy ndarray of objects
        The returned array of siddons with a 2 arrays of raveled voxel
        indices and intersection lengths
    nPixels : (3) array_like
        The number of voxels in the X, Y, and Z dimensions associated with the
        object array

    Returns
    -------
    [(nElem) numpy array, (nElem) numpy array]
        the returned array is of shape (...) with 4 arrays of the unraveled voxle
        indices and intersection lengths of the ray.
    """
    for ray_idx, ray in np.ndenumerate(sdlist):
        if ray != None:
            sdlist[ray_idx] = ray_unravel(ray,nPixels)
            
    return sdlist


def ray_ravel(ray,nPixels):
    """
    Ravels a single unraveled ray in an object array created by Siddons.

    Parameters
    ----------
    ray : (4) (nElem) numpy arrays
        A list contating 3 np.arrays of unraveled voxel indices and an np.array
        of intersection lengths all with size of nElems 
    nPixels : (3) array_like
        The number of voxels in the X, Y, and Z dimensions associated with the
        object array

    Returns
    -------
    [(nElem) numpy array, (nElem) numpy array]
        returns a list with 2 numpy arrays of raveled voxel index and
        coinciding intersection lengths with sizes of nElems
    """
    return [np.ravel_multi_index((ray[0],ray[1],ray[2]),nPixels),ray[3]]


def ray_unravel(ray,nPixels):
    """
    Unravels a single ray in an object array created by Siddons.

    Parameters
    ----------
    ray : (2) (nElem) numpy arrays
        A list containing an np.array of raveled voxel indices and an np.array
        of intersection lengths all with size of nElems 
    nPixels : (3) array_like
        The number of voxels in the X, Y, and Z dimensions

    Returns
    -------
    [(nElem) numpy array, (nElem) numpy array]
        the returned array is of shape (...) with 4 lists of the unraveled
        pixel indices and intersection length
    """

    return list(np.unravel_index(ray[0], nPixels) + (ray[1],))


def siddons(src, trg, nPixels=128, dPixels=1.0, origin=0.0,\
            ravel=False, flat=False, epsilon = 1e-5):
    """
    An implementation of Siddon's algorithm (Med. Phys., 12, 252-255) for 
    computing the intersection lengths of a line specified by the coordinates
    "source" and "target" with a (X,Y,Z) grid of voxels

    Parameters
    ----------
    src : (3) or (...,3) array_like
        The coordinates of the source position. ... shape must equal target or 
        be equal to one.
    trg : (3) or (...,3) array_like
        The coordinates of the target position. (Target may have multiple 
        postions and source 1 postiion)
    nPixels : int or (3) array_like
        The number of voxels in the X, Y, and Z dimensions For 2d grids set one
        of the dimesniosn equal to 1. Default is 512
    dPixels : float or (3) array_like
        The sampling interval in the X, Y, and Z dimensions. Default is 1.0
    cPixels : float or (3) array_like
        The center location in the X, Y, and Z dimensions. Default is 0.0
    ravel : bool, optional
        Stores the pixel values as either a raveled array index array (a
        single integer value representing all three dimesions) or an unraveled
        array index (3 arrays corresponding to the X, Y, and Z indices)
        dimensional array. Default is unravled
    flat : bool, optional
        Stores the data in two np.arrays with raveled pixel indices and
        intersection lengths. This format loses spatial encoding implict in trg
        and srcs arrays, but improves performance in accessing indices and 
        intersection lengths. Applicable in averaging across multiple rays to
        calculate solid angles

    Returns
    -------
    (...) numpy ndarray of objects
        the returned array is of shape (...) with 2 lists of the raveled pixel
        index and intersection lengths of the ray between the src and trg 
    """
    
    
    #Machine precision

    
    #Creates the grid of voxels
    g = vir.Grid3d(nPixels=nPixels,dPixels=dPixels,origin=origin, dtype=np.float32)

    #Converts and assigns paramters to approprate data types
    trg = np.array(trg, dtype=np.float32)
    if trg.ndim == 1:
        trg = trg[np.newaxis,:]

    src = np.array(src, dtype=np.float32)
    if src.ndim == 1:
        src = src[np.newaxis,:]

    #Number of rays
    Rays = np.empty(np.shape(trg)[:-1], dtype=object)

    #Creates flat arrays if required. Numpy array memory is overallocated and
    #filled within loop. This avoids np.concatneation requirement to locate and
    #allocate memory at each iteration. The size is reduced at the end
    if flat == True:
        f0 = 0
        flat_len = np.zeros(3*Rays.size*np.max(nPixels),dtype=np.float32)
        flat_ind = np.zeros(3*Rays.size*np.max(nPixels),dtype=int)
        
    #Creates the array of grid boundaries
    p0 = np.array([g.Xb[0],g.Yb[0],g.Zb[0]], dtype=np.float32)
    pN = np.array([g.Xb[-1],g.Yb[-1],g.Zb[-1]], dtype=np.float32)

    #Calculates deltas between target and source and Euclidean distance 
    dST  = trg - src #(nRays, 3)
    distance = np.linalg.norm(dST, axis=-1) #(nRays)

    #Calculate the parametric values of the intersections of the ray with the 
    #first and last grid lines. In the case of paralle lines where dST is equal
    #to 0.0, dST is set to epislon for alpha0 and alphaN caluclations. This will
    #allow ensure ray the grid intersection check works 
    with np.errstate(divide='ignore'):
        alpha0 = np.where(dST != 0.0, (p0-src)/dST, (p0-src)/epsilon)
        alphaN = np.where(dST != 0.0, (pN-src)/dST, (pN-src)/epsilon)
            
    #Calculate alpha_min and alpah max, which is either the parametric value of
    #the intersection where the line of interest enters or leaves the grid, or
    #0.0 if the trg is inside the grid.
    m_min = np.zeros(Rays.shape + (4,), dtype=np.float32)
    m_max = np.ones(Rays.shape + (4,), dtype=np.float32)
    
    m_min[...,1:][alpha0 < alphaN] = alpha0[alpha0 < alphaN] 
    m_max[...,1:][alpha0 < alphaN] = alphaN[alpha0 < alphaN] 

    m_min[...,1:][alpha0 > alphaN] = alphaN[alpha0 > alphaN] 
    m_max[...,1:][alpha0 > alphaN] = alpha0[alpha0 > alphaN] 

    alpha_min = np.max(m_min, axis=-1)
    alpha_max = np.min(m_max, axis=-1)
    alpha_bounds = np.stack([alpha_min,alpha_max], axis = -1)

    #Calculates the index bounds
    #Rounds values to an index if it within epsilon (error due to floating point arithmitic)
    i0p = (g.nPixels+1) - (pN-alpha_min[...,np.newaxis]*dST - src)/g.dPixels
    i0n = (g.nPixels+1) - (pN-alpha_max[...,np.newaxis]*dST - src)/g.dPixels
    iNp = (src + alpha_max[...,np.newaxis]*dST - p0)/g.dPixels
    iNn = (src + alpha_min[...,np.newaxis]*dST - p0)/g.dPixels

    i0p = np.where(np.isclose(i0p, np.round(i0p), atol=epsilon), np.round(i0p),  np.floor(i0p)).astype(int)
    i0n = np.where(np.isclose(i0n, np.round(i0n), atol=epsilon), np.round(i0n),  np.floor(i0n)).astype(int)
    iNp = np.where(np.isclose(iNp, np.round(iNp), atol=epsilon), np.round(iNp),  np.ceil(iNp)).astype(int)
    iNn = np.where(np.isclose(iNn, np.round(iNn), atol=epsilon), np.round(iNn),  np.ceil(iNn)).astype(int)
    
    i0 = np.where(dST > 0.0, i0p, i0n)
    iN = np.where(dST > 0.0, iNp, iNn)    
    
    #Loops through the rays intersecting the source and target
    for ray_idx, ray in np.ndenumerate(Rays):
        
        #If alpha_max <= alpha_min, then the ray doesn't pass through the grid.
        if alpha_bounds[ray_idx + (1,)] > alpha_bounds[ray_idx + (0,)] and \
           not np.isclose(alpha_bounds[ray_idx + (1,)], alpha_bounds[ray_idx + (0,)], atol=epsilon):

            idxX = ray_idx + (0,)
            idxY = ray_idx + (1,)
            idxZ = ray_idx + (2,)

            #Compute the alpha values of the intersections of the line with all
            #the relevant planes in the grid.
            #Does not store them in ascending order since this not a critiera for
            #np.sort. There may be potential performance increases with pre-sorting
            #and using a different sorting algorithm
            if dST[idxX] != 0.0:
                X_alpha = (g.Xb[i0[idxX]:iN[idxX]] - src[idxX])/dST[idxX]
            else:
                X_alpha = np.array([])
                
            if dST[idxY] != 0.0:
                Y_alpha = (g.Yb[i0[idxY]:iN[idxY]] - src[idxY])/dST[idxY]
            else:
                Y_alpha = np.array([])
                
            if dST[idxZ] != 0.0:
                Z_alpha = (g.Zb[i0[idxZ]:iN[idxZ]] - src[idxZ])/dST[idxZ]
            else:
                Z_alpha = np.array([])
            
            #Merges and sorts alphas
            Alpha = np.sort(np.concatenate([alpha_bounds[ray_idx], X_alpha, Y_alpha, Z_alpha]))

            if np.isclose(Alpha[0], Alpha[1]):
                Alpha = Alpha[1:]
            
            if np.isclose(Alpha[-1], Alpha[-2]):
                Alpha = Alpha[:-1]
            
            #Loops through the alphas and calculates pixel length and pixel index
            dAlpha = Alpha[1:] - Alpha[:-1]
            mAlpha = 0.5 * (Alpha[1:] + Alpha[:-1])
            
            x_ind = ((src[idxX] + mAlpha*dST[idxX] - g.Xb[0])/g.dX).astype(int)
            y_ind = ((src[idxY] + mAlpha*dST[idxY] - g.Yb[0])/g.dY).astype(int)
            z_ind = ((src[idxZ] + mAlpha*dST[idxZ] - g.Zb[0])/g.dZ).astype(int)
           
            length = (distance[ray_idx]*dAlpha).astype(np.float32)
            
            #Stores the index and intersection length in flat, raveled, or
            #unraveled form
            if flat == True:
                fN = f0 + length.size
                flat_ind[f0:fN] = np.ravel_multi_index((x_ind,y_ind,z_ind),g.nPixels)
                flat_len[f0:fN] = length
                f0 += length.size
                
            else:
                if ravel == True:
                    Rays[ray_idx] = [np.ravel_multi_index((x_ind,y_ind,z_ind),g.nPixels),length]
                else:
                    Rays[ray_idx] = [x_ind,y_ind,z_ind,length]
                
    
    #Returns results as a list for space efficiency or in an array for
    #numerical operation efficiency
    if flat == True:
        return flat_ind[flat_len>0],flat_len[flat_len>0]
    else:
        return Rays




def calc_grid_lines(X,Y,Z, x_size, y_size, z_size):    
    
    f_arr1 =  np.zeros(X+1, dtype=np.float32)
    x_lines = ctypes.c_void_p(f_arr1.ctypes.data)

    f_arr2 =  np.zeros(Y+1, dtype=np.float32)
    y_lines = ctypes.c_void_p(f_arr2.ctypes.data)

    #f_arr3 =  np.zeros(Z+1, dtype=np.float32)
    #z_lines = ctypes.c_void_p(f_arr3.ctypes.data)

    f_arr3 =  np.arange(Z+1, dtype=np.float32)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    z_lines = f_arr3.ctypes.data_as(c_float_p)


    c_test.calc_grid_lines(x_lines, y_lines, z_lines, \
                           ctypes.c_int(X), ctypes.c_int(Y), ctypes.c_int(Z),\
                           ctypes.c_float(x_size), ctypes.c_float(y_size), ctypes.c_float(z_size))
    
    print(f_arr1)
    print(f_arr2)
    print(f_arr3)


class VectI(ctypes.Structure):
     _fields_ = [("x", ctypes.c_int32),
                ("y", ctypes.c_int32),
                ("z", ctypes.c_int32)]


class VectF(ctypes.Structure):
     _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("z", ctypes.c_float)]


class VoxelLength(ctypes.Structure):
     _fields_ = [("idx", ctypes.c_int),
                ("length", ctypes.c_float)]


def siddon_c(src, trg, nPixels=128, dPixels=1.0):
    #Converts and assigns paramters to approprate data types
    src = np.array(src, dtype=np.float32)        
    trg = np.array(trg, dtype=np.float32)
    nPixels = vir.np_array(nPixels,3, dtype=np.int32)
    dPixels = vir.np_array(dPixels,3, dtype=np.float32)

    #Allocates memory for the arrays
    Xb =  np.empty(nPixels[0]+1, dtype=np.float32)
    Yb =  np.empty(nPixels[1]+1, dtype=np.float32)
    Zb =  np.empty(nPixels[2]+1, dtype=np.float32)

    X_alpha =  np.empty(nPixels[0]+1, dtype=np.float32)
    Y_alpha =  np.empty(nPixels[1]+1, dtype=np.float32)
    Z_alpha =  np.empty(nPixels[2]+1, dtype=np.float32)
    Alpha =  np.empty(2*(max(nPixels[0],nPixels[1],nPixels[2])+2)+1, dtype=np.float32)
    
    #Assigns c types pointers to variables
    c_float_p = ctypes.POINTER(ctypes.c_float)
    
    #Plane boundary pointers
    Xb_c = Xb.ctypes.data_as(c_float_p)
    Yb_c = Yb.ctypes.data_as(c_float_p)
    Zb_c = Zb.ctypes.data_as(c_float_p)

    #Alpha array pointers
    X_alpha_c = X_alpha.ctypes.data_as(c_float_p)
    Y_alpha_c = Y_alpha.ctypes.data_as(c_float_p)
    Z_alpha_c = Z_alpha.ctypes.data_as(c_float_p)
    Alpha_c = Alpha.ctypes.data_as(c_float_p)

    #Struct array pointers
    nPixels_c = VectI(ctypes.c_int32(nPixels[0]),ctypes.c_int32(nPixels[1]),ctypes.c_int32(nPixels[2]))
    dPixels_c = VectF(ctypes.c_float(dPixels[0]),ctypes.c_float(dPixels[1]),ctypes.c_float(dPixels[2]))
    src_c = VectF(ctypes.c_float(src[0]),ctypes.c_float(src[1]),ctypes.c_float(src[2]))
    trg_c = VectF(ctypes.c_float(trg[0]),ctypes.c_float(trg[1]),ctypes.c_float(trg[2]))

    rays_c = (VoxelLength*200)()
    
    #Calculates the grid lines
    c_test.calc_grid_lines(Xb_c, Yb_c, Zb_c, nPixels_c, dPixels_c)
 
    #
    c_test.siddons(src_c, trg_c, nPixels_c, dPixels_c, Xb_c, Yb_c, Zb_c,\
                   X_alpha_c, Y_alpha_c, Z_alpha_c, Alpha_c, \
                   ctypes.byref(rays_c))

    print("OKay")
    print(rays_c[0].idx, rays_c[0].length)
    print(rays_c[1].idx, rays_c[1].length)
    print(rays_c[2].idx, rays_c[2].length)
    print(rays_c[3].idx, rays_c[3].length)
    print(rays_c[4].idx, rays_c[4].length)
    print(rays_c[5].idx, rays_c[5].length)
    print(rays_c[6].idx, rays_c[6].length)
    



def forward_project_c(iArray, dArray, nPixels=128, dPixels=1.0):
    #Converts and assigns paramters to approprate data types
    nPixels = vir.np_array(nPixels,3, dtype=np.int32)
    dPixels = vir.np_array(dPixels,3, dtype=np.float32)

    #Struct array pointers
    nPixels_c = VectI(ctypes.c_int32(nPixels[0]),ctypes.c_int32(nPixels[1]),ctypes.c_int32(nPixels[2]))
    dPixels_c = VectF(ctypes.c_float(dPixels[0]),ctypes.c_float(dPixels[1]),ctypes.c_float(dPixels[2]))

    #Assigns c types pointers to variables
    c_float_p = ctypes.POINTER(ctypes.c_float)
    
    #Plane boundary pointers
    iArray_c = iArray.ctypes.data_as(c_float_p)
    dArray_c = dArray.ctypes.data_as(c_float_p)
        
    #Calculates the grid lines
    c_test.forward_project(iArray_c, dArray_c, nPixels_c, dPixels_c)
 

