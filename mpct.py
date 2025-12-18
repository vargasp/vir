#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:53:21 2022

@author: vargasp
"""

#import os, psutil
import multiprocessing as mp
import ctypes

import numpy as np
import vir.sys_mat.siddon as sd



"""
def ps_mem():
    return psutil.Process().memory_full_info().rss/(1024**3)

def ps_info(out='', m=[]):
    cpu = psutil.Process().cpu_num()
    pid = os.getpid()
    ppid = os.getppid()
    m.append(psutil.Process().memory_full_info().rss/(1024**3))
   
    print(out + f': CPU#: {cpu:3}, pid: {pid}, ppid: {ppid}, Mem RSS: {m[0]:.2f}GB', end='')
    for i in m[1:]:
        print(f', {i:.2f}GB', end='')
    print()
"""


c_int_p = ctypes.POINTER(ctypes.c_int)
c_float_p = ctypes.POINTER(ctypes.c_float)
c_double_p = ctypes.POINTER(ctypes.c_double)


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


def ctypes_vars(var):
    """
    Creates a ctypes object for an variety of data types.
    If the data is an np.array.
     - The original datatype will be converted to either float32 or int32.
     - The data will not be be copied.
     - ctypes object will point to the np.array data
     - Modifying the ctpyes data will change the np.array data
    If the data is a float or int
     - a cytpes object of the data will be created
     - The data will be copied
     - Modifying the ctypes object will not change the orginal python data
    
    Parameters
    ----------
    var : float, int, np.array
        Variable

    Returns
    -------
    var and ctypes var
        The orginal variable converted if necessay and ctypes pointer
    """
    
    # Checks if the variable is an np.array
    #if isinstance(var, (np.ndarray, np.generic) ):
    if isinstance(var, np.ndarray):

        # Changes the datatype from floatX to float32
        if isinstance(var.flat[0], np.float16) or isinstance(var.flat[0], np.float64):
            #print("Intstance 1 ")
            var = var.astype(np.float32)
            return var, var.ctypes.data_as(c_float_p)
            
        # Changes the datatype from intX to int32
        elif isinstance(var.flat[0], np.int8) or isinstance(var.flat[0], np.int16) \
            or isinstance(var.flat[0], np.int64):
            #print("Intstance 2 ", var.dtype)
            var = var.astype(np.int32)
            return var, var.ctypes.data_as(c_int_p)
            
        # Changes the datatype from uintX to int32
        elif isinstance(var.flat[0], np.uint8) or isinstance(var.flat[0], np.uint16) \
            or isinstance(var.flat[0], np.uint32) or isinstance(var.flat[0], np.uint64):
            #print("Intstance 3 ")
            var = var.astype(np.int32)
            return var, var.ctypes.data_as(c_int_p)
        
        # Create a ctypes object as a view into the original array (no copy)
        if isinstance(var.flat[0], np.float32): 
            #print("Intstance 4 ")

            return var, var.ctypes.data_as(c_float_p)
        
        elif isinstance(var.flat[0], np.int32):
            #print("Intstance 5 ")
            return var, var.ctypes.data_as(c_int_p)

        else:
            raise ValueError(var.dtype, " is not yet supported in mpct")

    # Create a ctypes object as of orginal data (copies)
    elif isinstance(var, float):
        return var, ctypes.c_float(var)
    
    # Create a ctypes object as of orginal data (copies)
    elif isinstance(var, int):
        return var, ctypes.c_int(var)
    # Create a ctypes object as of orginal data (copies)
    elif isinstance(var, np.integer):
        var = int(var)
        return var, ctypes.c_int(var)
    
    # Checks 
    else:
        raise ValueError(type(var),  " is not yet supported in mpct")


class UnraveledRay(ctypes.Structure):
     _fields_ = [("n", ctypes.c_int),
                 ("X", c_int_p),
                 ("Y", c_int_p),
                 ("Z", c_int_p),
                 ("L", c_float_p)]


class RaveledRay(ctypes.Structure):
     _fields_ = [("n", ctypes.c_int),
                 ("R", c_int_p),
                 ("L", c_float_p)]



class Coord(ctypes.Structure):
     _fields_ = [("x", ctypes.c_int),
                 ("y", ctypes.c_int),
                 ("z", ctypes.c_int)]

def ctypes_coord(x,y,z):
    return ctypes.byref(Coord(ctypes.c_int(x),\
                              ctypes.c_int(y),\
                              ctypes.c_int(z)))


def ctypes_unraveled_ray(n,X,Y,Z,L, byref=True):
    if byref:
        return ctypes.byref(UnraveledRay(ctypes.c_int(n),
                                       X.ctypes.data_as(c_int_p),\
                                       Y.ctypes.data_as(c_int_p),\
                                       Z.ctypes.data_as(c_int_p),\
                                       L.ctypes.data_as(c_float_p)))
    else:
        return UnraveledRay(ctypes.c_int(n),\
                          X.ctypes.data_as(c_int_p),\
                          Y.ctypes.data_as(c_int_p),\
                          Z.ctypes.data_as(c_int_p),\
                          L.ctypes.data_as(c_float_p))


def ctypes_raveled_ray(n,R,L, byref=True):
    if byref:
        return ctypes.byref(RaveledRay(ctypes.c_int(n),\
                                     R.ctypes.data_as(c_int_p),\
                                     L.ctypes.data_as(c_float_p)))
    else:
        return RaveledRay(ctypes.c_int(n),\
                        R.ctypes.data_as(c_int_p),\
                        L.ctypes.data_as(c_float_p))


def list_ctypes_object(sdlist,ravel=False,flat=True):
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
        count = sd.list_count_elem(sdlist).flatten()
        
        if ravel:
            sdlist_c = (RaveledRay*np.prod(sdlist.shape))()
        else:
            sdlist_c = (UnraveledRay*np.prod(sdlist.shape))()

    
        for ray_idx, ray in enumerate(sdlist.flatten()):
            sdlist_c[ray_idx].n =  ctypes_vars(count[ray_idx])[1]
 
            if ray != None and ravel==True:
                sdlist_c[ray_idx].R = ctypes_vars(ray[0])[1]
                sdlist_c[ray_idx].L = ctypes_vars(ray[1])[1]

            elif ray != None and ravel==False:
                sdlist_c[ray_idx].X = ctypes_vars(ray[0])[1]
                sdlist_c[ray_idx].Y = ctypes_vars(ray[1])[1]
                sdlist_c[ray_idx].Z = ctypes_vars(ray[2])[1]
                sdlist_c[ray_idx].L = ctypes_vars(ray[3])[1]
        sdlist_c = ctypes.byref(sdlist_c)

    else:
        count = sd.list_count_elem(sdlist)
        sdlist_c = np.empty(sdlist.shape, dtype=object)

        for ray_idx, ray in np.ndenumerate(sdlist):
            if ray != None and ravel==True:
                sdlist_c[ray_idx] = ctypes_raveled_ray(count[ray_idx],\
                                                          ray[0],ray[1])
                                                          
            elif ray != None and ravel==False:
                sdlist_c[ray_idx] = ctypes_unraveled_ray(count[ray_idx],\
                                                            ray[0],ray[1],\
                                                            ray[2],ray[3])
    return sdlist_c



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
