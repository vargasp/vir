#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:44:45 2020

@author: vargasp
"""

import numpy as np


#This block of code uses projection functions that rely on single core and in
#are soley in python. These are not optimzed for speed

def sd_f_proj_ray(phantom, ray):
    """
    Forward projects a single ray in an object array created by Siddons.

    Parameters
    ----------
    ray : (4) (nElem) numpy arrays
        A ray from a siddons object list with unraveled indices 
    phantom : (3) array_like
        Discretized numerical phantom

    Returns
    -------
    float
        The line integral summation of the ray through the phantom
    """
    return np.sum(phantom[(ray[0],ray[1],ray[2])] * ray[3])


def sd_b_proj_ray(phantom, ray, value):
    """
    Back projects a single ray in an object array created by Siddons.

    Parameters
    ----------
    ray : (4) (nElem) numpy arrays
        A ray from a siddons object list with unraveled indices 
    phantom : (3) array_like
        Discretized numerical phantom
    value : float
        The value to project along the ray

    Returns
    -------
    None
    
        Modifies the phantom parameter 
    """
    phantom[(ray[0],ray[1],ray[2])] += ray[3]*value
    

def sd_f_proj(phantom, sdlist, ravel=True, flat=True,sino_shape=None):
    """
    Forward projects all of the rays in an object array created by Siddons.

    Parameters
    ----------
    sdlist : (...) numpy ndarray of objects
        the returned array is of shape (...) with 2 lists of the raveled pixel
        index and intersection lengths or 4 lists of the unraveled pixel 
        indexes and intersection lengths from siddons 
    phantom : (3) array_like
        Discretized numerical phantom

    Returns
    -------
    float
        The line integral summation of the ray through the phantom
    """
    
    if sino_shape == None:
        sino_shape = sdlist.shape
        
    
    sino = np.zeros(sino_shape, dtype=np.float32)
    
    #Creates views of the data a 1d arrays for faster iteration    
    sino_flat = sino.ravel()


    if flat:
        phantom_flat = phantom.ravel()


        for i in range(sdlist[2].size-1):
            sIdx = sdlist[2][i]
            eIdx = sdlist[2][i+1]

            sino_flat[i] = np.sum(phantom_flat[sdlist[0][sIdx:eIdx]] \
                                  * sdlist[1][sIdx:eIdx]) 
        
        return sino

    #Creates views of the data a 1d arrays for faster iteration    
    sdlist_flat = sdlist.ravel()


    #Iterates through all of the lists    
    if ravel:
        phantom_flat = phantom.ravel()

        for i in range(sdlist_flat.size):
            if sdlist_flat[i] != None:
                sino_flat[i] = np.sum(phantom_flat[sdlist_flat[i][0]] * sdlist_flat[i][1])
        
    else:
        for i in range(sdlist_flat.size):
            if sdlist_flat[i] != None:
                sino_flat[i] = sd_f_proj_ray(phantom, sdlist_flat[i])
    
    
    return sino


def sd_b_proj(sino, sdlist, nPixels, ravel=True):
    
    phantom = np.zeros(nPixels, dtype=np.float32)
    
    #Creates views of the data a 1d arrays for faster iteration    
    sdlist_flat = sdlist.ravel()
    sino_flat = sino.ravel()

    
    #Iterates through all of the lists    
    if ravel:
        phantom_flat = phantom.ravel()

        for i in range(sdlist_flat.size):
            if sdlist_flat[i] != None:
                phantom_flat[sdlist_flat[i][0]] += sdlist_flat[i][1]*sino_flat[i]
            
    else:
        for i in range(sdlist_flat.size):
            if sdlist_flat[i] != None:
                phantom[(sdlist_flat[i][0],sdlist_flat[i][1],sdlist_flat[i][2])]\
                        += sdlist_flat[i][3]*sino_flat[i]
    
    
    """
    #Iterates through all of the lists    
    for ray_idx, ray in np.ndenumerate(sdlist):
        if ray != None:
            phantom[(ray[0],ray[1],ray[2])] += ray[3]*sino[ray_idx]
    """
    
    return phantom





