#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:53:21 2022

@author: vargasp
"""

import multiprocessing as mp
import numpy as np

import ctypes


class Ray(ctypes.Structure):
     _fields_ = [("n", ctypes.c_int),
                ("X", ctypes.POINTER(ctypes.c_int)),
                ("Y", ctypes.POINTER(ctypes.c_int)),
                ("Z", ctypes.POINTER(ctypes.c_int)),
                ("L", ctypes.POINTER(ctypes.c_float))]


class Coord(ctypes.Structure):
     _fields_ = [("x", ctypes.c_int),
                ("y", ctypes.c_int),
                ("z", ctypes.c_int)]


def c_int(var):
    """
    Creates a ctypes data type for an integer. (reduces importing ctypes in
    other modules)
    
    Parameters
    ----------
    var : int 
        Integer variable

    Returns
    -------
    ctypes c_int
        Int ctypes data type
    """
    return ctypes.c_int(var)


def c_float(var):
    """
    Creates a ctypes data type for an float. (reduces importing ctypes in
    other modules)
    
    Parameters
    ----------
    var : float 
        Float variable

    Returns
    -------
    ctypes c_int
        Float ctypes data type
    """
    return ctypes.c_float(var)


def ctypes_coord(x,y,z):
    return ctypes.byref(Coord(ctypes.c_int(x),ctypes.c_int(y),ctypes.c_int(z)))

def ctypes_ray(n,X,Y,Z,L, byref=True):
    if byref:
        return ctypes.byref(Ray(ctypes.c_int(n), np.ctypeslib.as_ctypes(X),\
                                np.ctypeslib.as_ctypes(Y), np.ctypeslib.as_ctypes(Z), \
                                np.ctypeslib.as_ctypes(L)))
    else:
        return Ray(ctypes.c_int(n), np.ctypeslib.as_ctypes(X),np.ctypeslib.as_ctypes(Y), \
                   np.ctypeslib.as_ctypes(Z), np.ctypeslib.as_ctypes(L))


def ctypes_vars(var):
    """
    Creates a ctypes data type for an variety of data types and converts the
    orginal arrays if ncessary
    
    Parameters
    ----------
    var : float, int, np.array
        Variable

    Returns
    -------
    var and ctypes var
        The orginal variable converted if necessay and ctypes pointer
    """
    if isinstance(var, (np.ndarray, np.generic) ):
        if isinstance(var.flat[0], np.float16) or isinstance(var.flat[0], np.float64) \
            or isinstance(var.flat[0], np.float128): 
            var = var.astype(np.float32)
            
        elif isinstance(var.flat[0], np.int8) or isinstance(var.flat[0], np.int16) \
            or isinstance(var.flat[0], np.int64):
            var = var.astype(np.int32)
            
        elif isinstance(var.flat[0], np.uint8) or isinstance(var.flat[0], np.uint16) \
            or isinstance(var.flat[0], np.uint32) or isinstance(var.flat[0], np.uint64):
            var = var.astype(np.int32)
        
        if isinstance(var.flat[0], np.float32) or isinstance(var.flat[0], np.int32):
            return var, np.ctypeslib.as_ctypes(var)
        else:
            raise ValueError(var.dtype, " is not yet supported in mpct")
                
    elif isinstance(var, float):
        return var, ctypes.c_float(var)
    
    elif isinstance(var, int):
        return var, ctypes.c_int(var)
    
    else:
        raise ValueError(type(var),  " is not yet supported in mpct")


def init_shared_array(arr, shape, dtype=float, l=None):
    global shared_arr

    shared_arr = {}
    shared_arr['arr'] = arr
    shared_arr['shape'] = shape  
    shared_arr['dtype'] = dtype
    shared_arr['lock'] = l


def shared_to_numpy(arr_shared, shape, dtype=float):
    """
    Create a NumPy array wrapper from a shared memory buffer with a given
    dtype and shape. No copy is involved, the array reflects the underlying
    shared buffer.
    
    Parameters
    ----------
    arr_shared : shared array 
        A ctypes array allocated from shared memory
    shape : int or (n) array_like
        The lengths of the corresponding array dimensions
    dtype : float, optional
        the data type of the new array

    Returns
    -------
    numpy ndarray 
        The numpy array wrapper
    """
    
    # Wrap shared array as an numpy array so we can easily manipulates its data.
    return np.frombuffer(arr_shared, dtype=dtype).reshape(shape)


def create_shared_array(shape, lock=True, dtype=float):
    """
    Create a new shared array. Return the shared array pointer, and a NumPy
    array wrapper for it. Note that the buffer values are not initialized.
    
    Parameters
    ----------
    shape : int or (n) array_like
        The lengths of the corresponding array dimensions
    lock : bool 
        Creates a locked or unlocked shared array. Read only arrays may be
        unlocked. Writeable arrays should be locked to allow synchronization
    dtype : float, optional
        the data type of the new array

    Returns
    -------
    ctypes array, numpy ndarray
    or id lock = True
    ctypes array, numpy ndarray, lock object l
        Returns the ctypes shared array, numpy wrapper and lock of for
        synchronization if lock = true
    """
    
    dtype = np.dtype(dtype)

    # Get a ctype type from the NumPy dtype.
    cdtype = np.ctypeslib.as_ctypes_type(dtype)

    # Create the RawArray instance.
    if lock:
        # Create a shared array of double precision with a lock.
        lock_obj = mp.Lock()
        arr_shared = mp.Array(cdtype, int(np.prod(shape)), lock=lock_obj)

        # Get a NumPy array wrapper
        arr_np = shared_to_numpy(arr_shared.get_obj(), shape, dtype=dtype)

        return arr_shared, arr_np, lock_obj

    else:
        # Create a shared array of double precision without a lock.
        arr_shared = mp.Array(cdtype, int(np.prod(shape)), lock=False)

        # Get a NumPy array wrapper
        arr_np = shared_to_numpy(arr_shared, shape, dtype=dtype)
    
        return arr_shared, arr_np
    
    
    
    
    
    
####DEPRECATED
def create_ctypes_array(arr):
    dtype = arr.dtype
    shape = arr.shape
    
    # Get a ctype type from the NumPy dtype.
    cdtype = np.ctypeslib.as_ctypes_type(dtype)
    
    #Converts only metadata. The array contents, the actual array data is
    #unchanged (not copied / modified / altered, ...).
    arr_c = np.ctypeslib.as_ctypes(arr)
    
    #Old method, not sure the main differnces
    #c_float_p = ctypes.POINTER(ctypes.c_float)
    #arr_c = arr.astype(np.float32).ctypes.data_as(c_float_p)
    
    #Wraps array in NumPy 
    arr_np = np.ctypeslib.as_array(arr_c, shape=shape)

    return arr_np, arr_c
