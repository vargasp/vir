#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:39:58 2024

@author: pvargas21
"""

import numpy as np


def geom_square(BinsX, distY, Angs):
    """
    Return the source and detector coordinates for equal-angular fanbeam,
    geometry on a square

    Parameters
    ----------
    BinsX : (nBins) numpy ndarray 
        X locations of the detectors
    distY : float
        Y location of the detectors.
    Angs : (nAngs) numpy ndarray 
        Angles of line of response of the detectors(radians)

    Returns
    -------
    srcs : TYPE
        DESCRIPTION.
    trgs : TYPE
        DESCRIPTION.

    """
    
    #The number of detectors and angles for each side array
    nBins = BinsX.size
    nAngs = Angs.size
    
    #Initializes the source and target postion arrays (nBins,nAngles,nSides,coord)
    srcs = np.zeros([nBins,nAngs,4,3])
    
    #Calculates the source position for the 4 arrays
    srcs[...,[0,1],[0,1]] = np.repeat(BinsX,nAngs*2).reshape((nBins,nAngs,2))
    srcs[...,[2,3],[0,1]] = np.repeat(-BinsX,nAngs*2).reshape((nBins,nAngs,2))       
    srcs[...,[0,1],[1,0]] = np.broadcast_to(-distY,(nBins,nAngs,2))
    srcs[...,[2,3],[1,0]] = np.broadcast_to(distY,(nBins,nAngs,2))
    srcs[...,2] = np.broadcast_to(0.0,(nBins,nAngs,4))

    #Calculates the target position for the 4 arrays
    #distY*5 will ensure target position doesn't fall within the fov
    trgs = srcs.copy()
    trgs[...,[0,1],[0,1]] += (np.cos(Angs)*distY*5)[...,np.newaxis]
    trgs[...,[2,3],[0,1]] -= (np.cos(Angs)*distY*5)[...,np.newaxis]
    trgs[...,[0,1],[1,0]] += (np.sin(Angs)*distY*5)[...,np.newaxis]
    trgs[...,[2,3],[1,0]] -= (np.sin(Angs)*distY*5)[...,np.newaxis]
    
    return srcs, trgs



def geom_circular(DetsY, Views, geom="par", src_iso=None, det_iso=None, DetsZ=None):
    """
    Return the source and detector coordinates for circular parallel beam
    geometry
    
    Parameters
    ----------
    DetsY : (nDet, nDetlets) or (nDets) numpy ndarray
        The array of detector positions relative to the isocener
    Views : (nViews, nViewLets) or (nViews) numpy ndarray
        The array of projections angles
    n : int float
        Distance greater than grid
    """
    if geom == "fan":
        if src_iso is None:
            raise ValueError("src_iso must be provided with geom=fan")
        if det_iso is None:
            raise ValueError("det_iso must be provided with geom=fan")
        if DetsZ is None:
            z_src = 0
            z_trg = 0
            y_trg = DetsY
        else:
            z_src = DetsZ
            z_trg = DetsZ
            y_trg = DetsY[:,np.newaxis]            

        y_src = 0.0
        
    elif geom == "cone":
        if src_iso is None:
            raise ValueError("src_iso must be provided with geom=cone")
        if det_iso is None:
            raise ValueError("det_iso must be provided with geom=cone")
        if DetsZ is None:
            raise ValueError("DetsZ must be provided with geom=cone")
        else:
            z_src = 0
            z_trg = DetsZ

        y_src = 0.0
        y_trg = DetsY[:,np.newaxis]
        
    elif geom == "par":
        if DetsZ is None:
            z_src = 0
            z_trg = 0
            y_src = DetsY
            y_trg = DetsY
        else:
            z_src = DetsZ
            z_trg = DetsZ
            y_src = DetsY[:,np.newaxis]
            y_trg = DetsY[:,np.newaxis]

        if det_iso is None:
            src_iso = np.max(DetsY) * 100
        if src_iso is None:
            det_iso = np.max(DetsY) * 100
        
    else:
        raise ValueError("Cannot process geom: ", geom)

    if DetsZ is None:
        Bins = Views.shape + DetsY.shape
    else:
        Bins = Views.shape + DetsY.shape + DetsZ.shape
        

    X_src = np.broadcast_to(-src_iso,Bins)
    X_trg = np.broadcast_to(det_iso,Bins)
    Y_src = np.broadcast_to(y_src,Bins)
    Y_trg = np.broadcast_to(y_trg,Bins)
    Z_src = np.broadcast_to(z_src,Bins)
    Z_trg = np.broadcast_to(z_trg,Bins)    
    
    src = np.stack([X_src, Y_src, Z_src],axis=-1)
    trg = np.stack([X_trg, Y_trg, Z_trg],axis=-1)

    for i, view in np.ndenumerate(Views):
        r_mat = np.array([[np.cos(view),np.sin(view)],[-np.sin(view), np.cos(view)]])
        
        src[i][...,:2] = np.matmul(np.stack([src[i][...,0],src[i][...,1]],axis=-1),r_mat)
        trg[i][...,:2] = np.matmul(np.stack([trg[i][...,0],trg[i][...,1]],axis=-1),r_mat)

    return (src, trg)



