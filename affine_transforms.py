# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:30:07 2023

@author: vargasp
"""

import numpy as np

from scipy.spatial import transform
from scipy.ndimage import map_coordinates

"""
Affine Tranformation Functions:
-------------------------------
"""

def rankIdn(A, rank):
    """
    Changes the rank of an affine tranformation matrix. Primarily used for
    matrix multiplicaiton operations. !!!DOES NOT MAP ALL ELEMENTS!!!

    Parameters
    ----------
    A : (n,m) np.array
        An affine transformation matrix
    rank : int
        The rank of the new affine transformation matrix

    Returns
    -------
    (rank, rank) np.array
        The new affine tranformation matrix or rank "rank"
    """
    n,m = A.shape
    
    if rank > n or rank >m:
        I = np.identity(rank)
        n,m  = min(n,rank), min(m,rank)
        I[:n,:m] = A[:n,:m]
        return I
    else:
        return A[:rank,:rank]


def transMat(coords,rank=None):
    """
    Creates an affine translation matrix for translation

    Parameters
    ----------
    coords : float or arraylike
        The translation vector 
    rank : int, optional
        The rank of the matrix. The default is "None" which is the lowest rank

    Returns
    -------
    T : (rank, rank) np.array
        The translation matrix in format: for coords = x; (x,y); and (x,y,z)
        |1 x| if x   |1 0 x| if (x,y)   |1 0 0 x| if (x,y,z)
        |0 1|        |0 1 y|            |0 1 0 y|
                     |0 0 1|            |0 0 1 z|
                                        |0 0 0 1|
    """
    #Calculates the dimension of the coords vector
    coords = np.array(coords)
    n = coords.size
    
    #Determines the rank of the matrix
    if rank is None:
        rank = n+1
    
    #Creates the tranlation matrix
    T = np.identity(rank)
    T[:n,-1] = coords
    
    return T


def rotateMat(angs, center=None, seq='XYZ', extrinsic=True, rank=2):
    """
    Creates an affine translation matrix for rotation

    Parameters
    ----------
    angs : float or arraylike
        The rotation vector.
    center : arraylike, optional
        The center of rotation location. The default is "None" which
        corresponds to the origin
    seq : TYPE, optional
        DESCRIPTION. The default is 'XYZ'.
    extrinsic : TYPE, optional
        DESCRIPTION. The default is True.
    rank : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    R : (rank, rank) np.array
        The rotation matrix in format: for angs = x; (0,x); and (0,0,x)
        |cos(x) -sin(x)| if x   |1 0 x| if (0,x)   |1 0 0 x| if (0,0,x)
        |sin(x)  cos(x)|        |0 1 y|            |0 1 0 y|
                                |0 0 1|            |0 0 1 z|
                                        |0 0 0 1|

    """

    #Converts angs to an np.array and calcualtes the number of angles
    angs = np.array(angs)
    n = angs.size
    
    #If more than one rotation is provided R must be at least rank 3 
    #Rank must be 1 more than the number of translation or have min of 3
    if n > 1: rank = max(rank, 3)    
    if center is None: rank = max(rank, 3)
    rank = max(rank, np.array(center).size + 1)


    #If the angle is intrinsic lower the sequence 
    if not extrinsic:
        seq = str.lower(seq)
    
    #Match the number of dimensions in the sequence to the number of angles
    seq = seq[(3-n):]
    
    #Calcuatd a 3x3 rotation matrix (centered at the 0,0)
    R = transform.Rotation.from_euler(seq, angs, degrees=False)
    R = R.as_matrix().squeeze()
    R = rankIdn(R, rank)

    #Returns the R matrix or changes the center of rotation if center is provided
    if center is None:
        return R
    else:
        T = transMat(center, rank=rank)
        
        return  T @ R @ np.linalg.inv(T)


def scaleMat(scale,center=None, rank=None):
    """
    Creates an affine translation matrix for scaling

    Parameters
    ----------
    scale : float or arraylike
        The sclaing vector. Values < 1 shink the image, while vales > 1 expand
        the image
    rank : int, optional
        The rank of the matrix. The default is "None" which is the lowest rank

    Returns
    -------
    T : (rank, rank) np.array
        The translation matrix in format: for coords = x; (x,y); and (x,y,z)
        |x 0| if x   |x 0 0| if (x,y)   |x 0 0 0| if (x,y,z)
        |0 1|        |0 y 0|            |0 y 0 0|
                     |0 0 1|            |0 0 z 0|
                                        |0 0 0 1|
    """
    
    #Converts scale vector to an np.array
    scale = np.array(scale)

    #Calculates the dimension of the coords vector
    if center is not None:
        center = np.array(center)
        center *= (1.0 - scale)

    #Increases the rank by one
    scale = np.append(scale,1)    
    
    #Determines the rank of the matrix
    if rank is None:
        rank = scale.size
    
    #Creates the tranlation matrix
    T = transMat(center, rank=rank)
    
    return T @ np.diag(scale)


def coords_array(shape,ones=False):
    
    if len(shape) == 1:
        coords = np.mgrid[:shape[0]]
    elif len(shape) == 2:
        coords = np.mgrid[:shape[0],:shape[1]]
    elif len(shape) == 3:
        coords = np.mgrid[:shape[0],:shape[1],:shape[2]]
    else:
        print("Dimensions must be between 1 and 3")

    if ones:
        coords = np.concatenate([coords,np.ones(shape)[np.newaxis,...]], axis = 0, dtype=float)
    else:
        coords = np.stack(coords, axis = 0, dtype=float)

    coords = np.moveaxis(coords,0, -2)
    coords = np.ascontiguousarray(coords)

    return coords



def coords_transform(arr, coords):
    coords = np.moveaxis(coords,-2, 0)[:-1,...]
    coords = np.ascontiguousarray(coords)
    
    return map_coordinates(arr, coords, order=1, mode='constant', cval=0.0)




