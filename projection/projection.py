#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:44:45 2020

@author: vargasp
"""
import itertools
import multiprocessing as mp
import numpy as np

import vir.analytic_geom as ag
import vir.mpct as mpct


import vir.siddon as sd
import ctypes

try:
    proj_so = ctypes.CDLL(__file__.rsplit("/",1)[0] + "/projection_c.so")
except:
    print("Warning. Projection shared object libray not present.")



#C Files
def sd_f_proj_c(phantom, sino, sdlist_c, flat=True):
    sino, sino_c = mpct.ctypes_vars(sino)
    phantom, phantom_c = mpct.ctypes_vars(phantom)

    nX, nY, nZ = phantom.shape
    dims_c = mpct.ctypes_coord(nX,nY,nZ)

    if flat:
        proj_so.forward_proj_rays_u_struct(phantom_c, sino_c, sdlist_c, \
                                   dims_c, mpct.c_int(sino.size))
    else:
        for ray_idx, ray_c in np.ndenumerate(sdlist_c):
            if ray_c != None:   
                sinoElemIdx = mpct.c_int(np.ravel_multi_index(ray_idx,sino.shape))
                proj_so.forward_proj_ray_u_struct(phantom_c, sino_c, ray_c, \
                                           dims_c, sinoElemIdx)
                

def sd_b_proj_c(phantom, sino, sdlist_c, flat=True):
    sino, sino_c = mpct.ctypes_vars(sino)
    phantom, phantom_c = mpct.ctypes_vars(phantom)

    nX, nY, nZ = phantom.shape
    dims_c = mpct.ctypes_coord(nX,nY,nZ)

    if flat:
        proj_so.back_proj_rays_u_struct(phantom_c, sino_c, sdlist_c, \
                                   dims_c, mpct.c_int(sino.size))
    else:
        for ray_idx, ray_c in np.ndenumerate(sdlist_c):
            if ray_c != None:
                sinoElemIdx = mpct.c_int(np.ravel_multi_index(ray_idx,sino.shape))
                proj_so.back_proj_ray_u_struct(phantom_c, sino_c, ray_c, \
                            dims_c, sinoElemIdx)
  

def sd_f_proj_t_c(phantom, sino, sdlist_c, sBinsY, sBinsZ, flat=True): 
    phantom, phantom_c = mpct.ctypes_vars(phantom)

    nBinsY = sBinsY[1] - sBinsY[0]
    nBinsZ = sBinsZ[1] - sBinsZ[0]

    nX, nY, nZ = phantom.shape
    dims_c = mpct.ctypes_coord(nX,nY,nZ)
    bins0_c = mpct.ctypes_coord(0,sBinsY[0],sBinsZ[0])
    binsN_c = mpct.ctypes_coord(0,sBinsY[1],sBinsZ[1])

    if flat:
        sino, sino_c =  mpct.ctypes_vars(sino)
        nRays_c = mpct.c_int(int(sino.size/nBinsY/nBinsZ))

        proj_so.forward_proj_rays_t(phantom_c, sino_c, sdlist_c,\
                                         dims_c, bins0_c, \
                                         binsN_c, nRays_c)
    else:
        sino_trans = np.zeros((nBinsY, nBinsZ), dtype=np.float32)
        sino_trans, sino_trans_c = mpct.ctypes_vars(sino_trans)

        for ray_idx, ray_c in np.ndenumerate(sdlist_c):
            if ray_c != None:
                sino_trans[:] = 0.0

                proj_so.forward_proj_ray_t(phantom_c, sino_trans_c, ray_c,\
                                 dims_c, bins0_c, binsN_c)

                sino[ray_idx] = sino_trans


def sd_b_proj_t_c(phantom, sino, sdlist_c, sBinsY, sBinsZ, flat=True): 
    phantom, phantom_c = mpct.ctypes_vars(phantom)

    nBinsY = sBinsY[1] - sBinsY[0]
    nBinsZ = sBinsZ[1] - sBinsZ[0]

    nX, nY, nZ = phantom.shape
    dims_c = mpct.ctypes_coord(nX,nY,nZ)
    bins0_c = mpct.ctypes_coord(0,sBinsY[0],sBinsZ[0])
    binsN_c = mpct.ctypes_coord(0,sBinsY[1],sBinsZ[1])

    if flat:
        sino, sino_c = mpct.ctypes_vars(sino)
        nRays_c = mpct.c_int(int(sino.size/nBinsY/nBinsZ))

        proj_so.back_proj_rays_t(phantom_c, sino_c, sdlist_c,\
                                         dims_c, bins0_c, \
                                         binsN_c, nRays_c)
    else:
        sino_trans = np.zeros((nBinsY, nBinsZ), dtype=np.float32)
        sino_trans, sino_trans_c = mpct.ctypes_vars(sino_trans)

        for ray_idx, ray_c in np.ndenumerate(sdlist_c):
            if ray_c != None:
                sino_trans[:,:] = sino[ray_idx].astype(np.float32)

                proj_so.back_proj_ray_t(phantom_c, sino_trans_c, ray_c,\
                                 dims_c, bins0_c, binsN_c)


def sd_f_proj_s_c(phantom, sino, sdlist_c, sBinsY, sBinsZ, flat=True): 
    phantom, phantom_c = mpct.ctypes_vars(phantom)

    nBinsY = sBinsY[1] - sBinsY[0]
    nBinsZ = sBinsZ[1] - sBinsZ[0]

    nX, nY, nZ = phantom.shape
    dims_c = mpct.ctypes_coord(nX,nY,nZ)
    bins0_c = mpct.ctypes_coord(0,sBinsY[0],sBinsZ[0])
    binsN_c = mpct.ctypes_coord(0,sBinsY[1],sBinsZ[1])

    if flat:
        sino, sino_c =  mpct.ctypes_vars(sino)
        nRays_c = mpct.c_int(int(sino.size/nBinsY/nBinsZ/2))

        proj_so.forward_proj_rays_sym(phantom_c, sino_c, sdlist_c,\
                                         dims_c, bins0_c, \
                                         binsN_c, nRays_c)
    else:
        sino_trans = np.zeros((nBinsY, nBinsZ, 2), dtype=np.float32)
        sino_trans, sino_trans_c = mpct.ctypes_vars(sino_trans)

        for ray_idx, ray_c in np.ndenumerate(sdlist_c):
            if ray_c != None:
                sino_trans[:] = 0.0

                proj_so.forward_proj_ray_sym(phantom_c, sino_trans_c, ray_c,\
                                 dims_c, bins0_c, binsN_c)

                sino[ray_idx] = sino_trans


def sd_b_proj_s_c(phantom, sino, sdlist_c, sBinsY, sBinsZ, flat=True): 
    phantom, phantom_c = mpct.ctypes_vars(phantom)

    nBinsY = sBinsY[1] - sBinsY[0]
    nBinsZ = sBinsZ[1] - sBinsZ[0]

    nX, nY, nZ = phantom.shape
    dims_c = mpct.ctypes_coord(nX,nY,nZ)
    bins0_c = mpct.ctypes_coord(0,sBinsY[0],sBinsZ[0])
    binsN_c = mpct.ctypes_coord(0,sBinsY[1],sBinsZ[1])

    if flat:
        sino, sino_c = mpct.ctypes_vars(sino)
        nRays_c = mpct.c_int(int(sino.size/nBinsY/nBinsZ/2))
        
        proj_so.back_proj_rays_sym(phantom_c, sino_c, sdlist_c,\
                                         dims_c, bins0_c, \
                                         binsN_c, nRays_c)
    else:
        sino_trans = np.zeros((nBinsY, nBinsZ,2), dtype=np.float32)
        sino_trans, sino_trans_c = mpct.ctypes_vars(sino_trans)

        for ray_idx, ray_c in np.ndenumerate(sdlist_c):
            if ray_c != None:
                sino_trans[:,:,:] = sino[ray_idx].astype(np.float32)

                proj_so.backproj_ray_sym(phantom_c, sino_trans_c, ray_c,\
                                 dims_c, bins0_c, binsN_c)















def mp_sd_f_proj_ray(ray):
    if ray != None:
        phantom = mpct.shared_to_numpy(mpct.shared_arr['arr'], \
                                       mpct.shared_arr['shape'], \
                                       dtype=mpct.shared_arr['dtype'])
        
        return np.sum(phantom[(ray[0],ray[1],ray[2])] * ray[3])
    else:
        return 0.0;


def mp_sd_f_proj(phantom, sdlist, cpus=mp.cpu_count()-1):
    nPixels = phantom.shape
    dtype = phantom.dtype

    # Create a shared arrays of double precision with a lock.
    # WARNING!!! Array should be read only  
    phantom_shrd, phantom_shrd_np = mpct.create_shared_array(nPixels, lock=False,dtype=dtype)

    # Copy data to shared array.
    np.copyto(phantom_shrd_np, phantom)
    
    # Start the process pool and do the computation.
    # Here we pass phantom, shape, and dtype to the initializer of each worker.
    # (Because shape & dtype are not a shared variables, they will be copied to each
    # child process.)
    with mp.Pool(processes=cpus, initializer=mpct.init_shared_array, \
                 initargs=(phantom_shrd, nPixels, dtype)) as pool:
        sino = pool.map(mp_sd_f_proj_ray, sdlist.flatten())

    return np.array(sino).reshape(sdlist.shape)
        

def mp_sd_b_proj_ray(args):
    value, ray = args

    if ray != None:
        phantom = mpct.shared_to_numpy(mpct.shared_arr['arr'], \
                                       mpct.shared_arr['shape'], \
                                       dtype=mpct.shared_arr['dtype'])
            
        mpct.shared_arr['lock'].acquire()
        phantom[(ray[0],ray[1],ray[2])] += ray[3]*value
        mpct.shared_arr['lock'].release()


def mp_sd_b_proj(sino, sdlist, nPixels,cpus=mp.cpu_count()-1):
    phantom = np.zeros(nPixels, dtype=np.float32)
    dtype = phantom.dtype
    
    # Create a shared arrays of double precision with a lock.
    phantom_shrd, phantom_shrd_np, l = mpct.create_shared_array(nPixels, lock=True,dtype=dtype)
    
    # Copy data to shared array.
    np.copyto(phantom_shrd_np, phantom)
    
    # Start the process pool and do the computation.
    # Here we pass phantom, shape, and dtype to the initializer of each worker.
    # (Because shape & dtype are not a shared variables, they will be copied to each
    # child process.)
    with mp.Pool(processes=cpus, initializer=mpct.init_shared_array, \
                 initargs=(phantom_shrd.get_obj(), nPixels, dtype, l)) as pool:
        pool.map(mp_sd_b_proj_ray, zip(sino.flatten(),sdlist.flatten()))

    return phantom_shrd_np


def mpc_sd_f_proj_func(ray):
    
    if ray != None:
        sinoVal_c = ctypes.c_float(0.0)
        sinoVal_cp = ctypes.pointer(sinoVal_c)
        
        phantom = mpct.shared_to_numpy(mpct.shared_arr['arr'], \
                                       mpct.shared_arr['shape'], \
                                       dtype=mpct.shared_arr['dtype'])
        
        phantom_c = mpct.ctypes_vars(phantom)[1]
        nX, nY, nZ = mpct.shared_arr['shape']
        dims_c = mpct.ctypes_coord(nX,nY,nZ)
        
        ray_c = mpct.ctypes_ray(ray[0].size,ray[0].astype(np.int32),ray[1].astype(np.int32),\
                                ray[2].astype(np.int32),ray[3].astype(np.float32))

        proj_so.forward_proj_ray_u_struct(phantom_c, sinoVal_cp, ray_c, dims_c, 0)
        return sinoVal_c.value
    else:
        return 0.0;


def mpc_sd_f_proj(phantom, sdlist, cpus=mp.cpu_count()-1):
    nPixels = phantom.shape
    dtype = phantom.dtype

    # Create a shared arrays of double precision with a lock.
    # WARNING!!! Array should be read only  
    phantom_shrd, phantom_shrd_np = mpct.create_shared_array(nPixels, lock=False,dtype=dtype)

    # Copy data to shared array.
    np.copyto(phantom_shrd_np, phantom)
    
    # Start the process pool and do the computation.
    # Here we pass phantom, shape, and dtype to the initializer of each worker.
    # (Because shape & dtype are not a shared variables, they will be copied to each
    # child process.)
    with mp.Pool(processes=cpus, initializer=mpct.init_shared_array, \
                 initargs=(phantom_shrd, nPixels, dtype)) as pool:
        sino = pool.map(mpc_sd_f_proj_func, sdlist.flatten())

    return np.array(sino).reshape(sdlist.shape)


def mpc_sd_b_proj_func(args):
    value, ray = args

    if ray != None:
        sinoVal_c = ctypes.c_float(value)
        sinoVal_cp = ctypes.pointer(sinoVal_c)
        
        phantom = mpct.shared_to_numpy(mpct.shared_arr['arr'], \
                                       mpct.shared_arr['shape'], \
                                       dtype=mpct.shared_arr['dtype'])

        phantom_temp = np.zeros(mpct.shared_arr['shape'], dtype = np.float32)
        phantom_temp, phantom_temp_c = mpct.ctypes_vars(phantom_temp)
        nX, nY, nZ = mpct.shared_arr['shape']
        dims_c = mpct.ctypes_coord(nX,nY,nZ)
        
        ray_c = mpct.ctypes_ray(ray[0].size,ray[0].astype(np.int32),ray[1].astype(np.int32),\
                                ray[2].astype(np.int32),ray[3].astype(np.float32))

        proj_so.back_proj_ray_u_struct(phantom_temp_c, sinoVal_cp, ray_c, dims_c, 0)
            
        mpct.shared_arr['lock'].acquire()
        phantom += phantom_temp
        mpct.shared_arr['lock'].release()


def mpc_sd_b_proj(sino, sdlist, nPixels, cpus=mp.cpu_count()-1):
    phantom = np.zeros(nPixels, dtype=np.float32)
    dtype = phantom.dtype
    
    # Create a shared arrays of double precision with a lock.
    phantom_shrd, phantom_shrd_np, l = mpct.create_shared_array(nPixels, lock=True,dtype=dtype)
    
    # Copy data to shared array.
    np.copyto(phantom_shrd_np, phantom)
    
    # Start the process pool and do the computation.
    # Here we pass phantom, shape, and dtype to the initializer of each worker.
    # (Because shape & dtype are not a shared variables, they will be copied to each
    # child process.)
    with mp.Pool(processes=cpus, initializer=mpct.init_shared_array, \
                 initargs=(phantom_shrd.get_obj(), nPixels, dtype, l)) as pool:
        pool.map(mpc_sd_b_proj_func, zip(sino.flatten(),sdlist.flatten()))

    return phantom_shrd_np


def mpc_sd_f_proj_t_func(args):
    ray, sBinsY, sBinsZ = args    
    
    nBinsY = sBinsY[1] - sBinsY[0]
    nBinsZ = sBinsZ[1] - sBinsZ[0]
    
    sino_trans = np.zeros((nBinsY, nBinsZ), dtype=np.float32)
        
    if ray != None:
        sino_trans, sino_trans_c = mpct.ctypes_vars(sino_trans)
        
        phantom = mpct.shared_to_numpy(mpct.shared_arr['arr'], \
                                       mpct.shared_arr['shape'], \
                                       dtype=mpct.shared_arr['dtype'])
        
        phantom_c = mpct.ctypes_vars(phantom)[1]
        nX, nY, nZ = mpct.shared_arr['shape']
        dims_c = mpct.ctypes_coord(nX,nY,nZ)
        bins0_c = mpct.ctypes_coord(0,sBinsY[0],sBinsZ[0])
        binsN_c = mpct.ctypes_coord(0,sBinsY[1],sBinsZ[1])
        
        ray_c = mpct.ctypes_ray(ray[0].size,ray[0].astype(np.int32),ray[1].astype(np.int32),\
                                ray[2].astype(np.int32),ray[3].astype(np.float32))

        proj_so.forward_proj_ray_t(phantom_c, sino_trans_c, ray_c,\
                                 dims_c, bins0_c, binsN_c)
    return sino_trans


def mpc_sd_f_proj_t(phantom, sdlist, sBinsY, sBinsZ, cpus=mp.cpu_count()-1):
    nPixels = phantom.shape
    dtype = phantom.dtype

    nBinsY = sBinsY[1] - sBinsY[0]
    nBinsZ = sBinsZ[1] - sBinsZ[0]

    phantom_shrd, phantom_shrd_np = mpct.create_shared_array(nPixels, lock=False,dtype=dtype)

    np.copyto(phantom_shrd_np, phantom)
    
    with mp.Pool(processes=cpus, initializer=mpct.init_shared_array, \
                 initargs=(phantom_shrd, nPixels, dtype)) as pool:
        sino = pool.map(mpc_sd_f_proj_t_func, zip(sdlist.flatten(),\
                                                  itertools.repeat(sBinsY), \
                                                  itertools.repeat(sBinsZ)))

    return np.array(sino).reshape(sdlist.shape + (nBinsY,nBinsZ,))


def mpc_sd_b_proj_t(sino, sdlist, nPixels, sBinsY, sBinsZ, cpus=mp.cpu_count()-1):
    phantom = np.zeros(nPixels, dtype=np.float32)
    
    dtype = phantom.dtype
    nBinsY = sBinsY[1] - sBinsY[0]
    nBinsZ = sBinsZ[1] - sBinsZ[0]
    
    # Create a shared arrays of double precision with a lock.
    phantom_shrd, phantom_shrd_np, l = mpct.create_shared_array(nPixels, lock=True,dtype=dtype)
    
    # Copy data to shared array.
    np.copyto(phantom_shrd_np, phantom)
    
    # Start the process pool and do the computation.
    # Here we pass phantom, shape, and dtype to the initializer of each worker.
    # (Because shape & dtype are not a shared variables, they will be copied to each
    # child process.)
    with mp.Pool(processes=cpus, initializer=mpct.init_shared_array, \
                 initargs=(phantom_shrd.get_obj(), nPixels, dtype, l)) as pool:
        pool.map(mpc_sd_b_proj_t_func, zip(sino.reshape(sdlist.size,nBinsY,nBinsZ),sdlist.flatten(),\
                                                  itertools.repeat(sBinsY), \
                                                  itertools.repeat(sBinsZ)))

    return phantom_shrd_np


def mpc_sd_b_proj_t_func(args):
    sino_trans, ray, sBinsY, sBinsZ = args    

    if ray != None:       
        bins0_c = mpct.ctypes_coord(0,sBinsY[0],sBinsZ[0])
        binsN_c = mpct.ctypes_coord(0,sBinsY[1],sBinsZ[1])
        
        sino_trans, sino_trans_c = mpct.ctypes_vars(sino_trans)
        
        phantom = mpct.shared_to_numpy(mpct.shared_arr['arr'], \
                                       mpct.shared_arr['shape'], \
                                       dtype=mpct.shared_arr['dtype'])

        phantom_temp = np.zeros(mpct.shared_arr['shape'], dtype = np.float32)
        phantom_temp, phantom_temp_c = mpct.ctypes_vars(phantom_temp)
        nX, nY, nZ = mpct.shared_arr['shape']
        dims_c = mpct.ctypes_coord(nX,nY,nZ)
        
        ray_c = mpct.ctypes_ray(ray[0].size,ray[0].astype(np.int32),ray[1].astype(np.int32),\
                                ray[2].astype(np.int32),ray[3].astype(np.float32))

        proj_so.back_proj_ray_t(phantom_temp_c, sino_trans_c, ray_c, dims_c, bins0_c, binsN_c)
            
        mpct.shared_arr['lock'].acquire()
        phantom += phantom_temp
        mpct.shared_arr['lock'].release()














def mpc_sd_f_proj_s(phantom, sdlist, sBinsY, sBinsZ, cpus=mp.cpu_count()-1):
    nPixels = phantom.shape
    dtype = phantom.dtype

    nBinsY = sBinsY[1] - sBinsY[0]
    nBinsZ = sBinsZ[1] - sBinsZ[0]

    phantom_shrd, phantom_shrd_np = mpct.create_shared_array(nPixels,lock=False,dtype=dtype)

    np.copyto(phantom_shrd_np, phantom)
    
    with mp.Pool(processes=cpus, initializer=mpct.init_shared_array, \
                 initargs=(phantom_shrd, nPixels, dtype)) as pool:
        sino = pool.map(mpc_sd_f_proj_s_func2, zip(sdlist,\
                                                  itertools.repeat(sBinsY), \
                                                  itertools.repeat(sBinsZ)))

    return np.array(sino).reshape(sdlist.shape + (nBinsY,nBinsZ,2,))



def mpc_sd_b_proj_s(sino, sdlist, nPixels, sBinsY, sBinsZ, cpus=mp.cpu_count()-1):
    phantom = np.zeros(nPixels, dtype=np.float32)  
    dtype = phantom.dtype
    
    phantom_shrd, phantom_shrd_np, l = mpct.create_shared_array(nPixels,lock=True,dtype=dtype)
    
    np.copyto(phantom_shrd_np, phantom)
    
    with mp.Pool(processes=cpus, initializer=mpct.init_shared_array, \
                 initargs=(phantom_shrd.get_obj(), nPixels, dtype, l)) as pool:
        pool.map(mpc_sd_b_proj_s_func2, zip(sino,sdlist,\
                                                  itertools.repeat(sBinsY), \
                                                  itertools.repeat(sBinsZ)))

    return phantom_shrd_np




def mpc_sd_b_proj_s_func(args):
    sino_trans, ray, sBinsY, sBinsZ = args    

    if ray != None:       
        bins0_c = mpct.ctypes_coord(0,sBinsY[0],sBinsZ[0])
        binsN_c = mpct.ctypes_coord(0,sBinsY[1],sBinsZ[1])
        
        sino_trans, sino_trans_c = mpct.ctypes_vars(sino_trans)
        
        phantom = mpct.shared_to_numpy(mpct.shared_arr['arr'], \
                                       mpct.shared_arr['shape'], \
                                       dtype=mpct.shared_arr['dtype'])

        phantom_temp = np.zeros(mpct.shared_arr['shape'], dtype = np.float32)
        phantom_temp, phantom_temp_c = mpct.ctypes_vars(phantom_temp)
        nX, nY, nZ = mpct.shared_arr['shape']
        dims_c = mpct.ctypes_coord(nX,nY,nZ)
        
        ray_c = mpct.ctypes_ray(ray[0].size,ray[0].astype(np.int32),ray[1].astype(np.int32),\
                                ray[2].astype(np.int32),ray[3].astype(np.float32))

        proj_so.backproj_ray_sym(phantom_temp_c, sino_trans_c, ray_c,\
                         dims_c, bins0_c, binsN_c)
            
        mpct.shared_arr['lock'].acquire()
        phantom += phantom_temp
        mpct.shared_arr['lock'].release()


def mpc_sd_f_proj_s_func(args):
    ray, sBinsY, sBinsZ = args    
    
    nBinsY = sBinsY[1] - sBinsY[0]
    nBinsZ = sBinsZ[1] - sBinsZ[0]
    
    sino_trans = np.zeros((nBinsY, nBinsZ,2), dtype=np.float32)
        
    if ray != None:
        sino_trans, sino_trans_c = mpct.ctypes_vars(sino_trans)
        
        phantom = mpct.shared_to_numpy(mpct.shared_arr['arr'], \
                                       mpct.shared_arr['shape'], \
                                       dtype=mpct.shared_arr['dtype'])
        
        phantom_c = mpct.ctypes_vars(phantom)[1]
        nX, nY, nZ = mpct.shared_arr['shape']
        dims_c = mpct.ctypes_coord(nX,nY,nZ)
        bins0_c = mpct.ctypes_coord(0,sBinsY[0],sBinsZ[0])
        binsN_c = mpct.ctypes_coord(0,sBinsY[1],sBinsZ[1])
        
        ray_c = mpct.ctypes_ray(ray[0].size,ray[0].astype(np.int32),ray[1].astype(np.int32),\
                                ray[2].astype(np.int32),ray[3].astype(np.float32))

        proj_so.forward_proj_ray_sym(phantom_c, sino_trans_c, ray_c,\
                                 dims_c, bins0_c, binsN_c)

    return sino_trans


def mpc_sd_f_proj_s_func2(args):
    sdlist, sBinsY, sBinsZ = args    
    
    bins0_c = mpct.ctypes_coord(0,sBinsY[0],sBinsZ[0])
    binsN_c = mpct.ctypes_coord(0,sBinsY[1],sBinsZ[1])
    
    nBinsY = sBinsY[1] - sBinsY[0]
    nBinsZ = sBinsZ[1] - sBinsZ[0]
    
    sino_trans = np.zeros(sdlist.shape + (nBinsY, nBinsZ,2), dtype=np.float32)
    sino_trans, sino_trans_c = mpct.ctypes_vars(sino_trans)

    phantom = mpct.shared_to_numpy(mpct.shared_arr['arr'], \
                                   mpct.shared_arr['shape'], \
                                   dtype=mpct.shared_arr['dtype'])

    phantom_c = mpct.ctypes_vars(phantom)[1]
    nX, nY, nZ = mpct.shared_arr['shape']
    dims_c = mpct.ctypes_coord(nX,nY,nZ)    
    nRays = int(np.prod(sdlist.shape))
    
    sdlist_c = (mpct.Ray*nRays)()
    for ray_idx, ray in enumerate(sdlist.flatten()):
        sdlist_c[ray_idx].n = ctypes.c_int(ray[0].size)
 
        if ray != None:
            sdlist_c[ray_idx].X = np.ctypeslib.as_ctypes(ray[0].astype(np.int32))
            sdlist_c[ray_idx].Y = np.ctypeslib.as_ctypes(ray[1].astype(np.int32))
            sdlist_c[ray_idx].Z = np.ctypeslib.as_ctypes(ray[2].astype(np.int32))
            sdlist_c[ray_idx].L = np.ctypeslib.as_ctypes(ray[3].astype(np.float32))
        
    sdlist_c = ctypes.byref(sdlist_c)

    proj_so.forward_proj_rays_sym(phantom_c, sino_trans_c, sdlist_c,\
                                 dims_c, bins0_c, binsN_c,nRays)

    return sino_trans


def mpc_sd_b_proj_s_func2(args):
    sino, sdlist, sBinsY, sBinsZ = args    

    bins0_c = mpct.ctypes_coord(0,sBinsY[0],sBinsZ[0])
    binsN_c = mpct.ctypes_coord(0,sBinsY[1],sBinsZ[1])
        
    sino, sino_c = mpct.ctypes_vars(sino)
        
    phantom = mpct.shared_to_numpy(mpct.shared_arr['arr'], \
                                   mpct.shared_arr['shape'], \
                                   dtype=mpct.shared_arr['dtype'])

    phantom_temp = np.zeros(mpct.shared_arr['shape'], dtype = np.float32)
    phantom_temp, phantom_temp_c = mpct.ctypes_vars(phantom_temp)
    nX, nY, nZ = mpct.shared_arr['shape']
    dims_c = mpct.ctypes_coord(nX,nY,nZ)
    nRays = int(np.prod(sdlist.shape))
    
    sdlist_c = (mpct.Ray*nRays)()
    for ray_idx, ray in enumerate(sdlist.flatten()):
        sdlist_c[ray_idx].n = ctypes.c_int(ray[0].size)
 
        if ray != None:
            sdlist_c[ray_idx].X = np.ctypeslib.as_ctypes(ray[0].astype(np.int32))
            sdlist_c[ray_idx].Y = np.ctypeslib.as_ctypes(ray[1].astype(np.int32))
            sdlist_c[ray_idx].Z = np.ctypeslib.as_ctypes(ray[2].astype(np.int32))
            sdlist_c[ray_idx].L = np.ctypeslib.as_ctypes(ray[3].astype(np.float32))
        
    sdlist_c = ctypes.byref(sdlist_c)
    

    proj_so.back_proj_rays_sym(phantom_temp_c, sino_c, sdlist_c,\
                         dims_c, bins0_c, binsN_c,nRays)
            
    mpct.shared_arr['lock'].acquire()
    phantom += phantom_temp
    mpct.shared_arr['lock'].release()








def mpc_sd_f_proj_4sym(phantom, sdlist, sBinsY, sBinsZ, cpus=mp.cpu_count()-1):
    nPixels = phantom.shape
    dtype = phantom.dtype

    nBinsY = sBinsY[1] - sBinsY[0]
    nBinsZ = sBinsZ[1] - sBinsZ[0]

    phantom_shrd, phantom_shrd_np = mpct.create_shared_array(nPixels,lock=False,dtype=dtype)

    np.copyto(phantom_shrd_np, phantom)
    
    with mp.Pool(processes=cpus, initializer=mpct.init_shared_array, \
                 initargs=(phantom_shrd, nPixels, dtype)) as pool:
        sino = pool.map(mpc_sd_f_proj_s_func4sym, zip(sdlist,\
                                                  itertools.repeat(sBinsY), \
                                                  itertools.repeat(sBinsZ)))

    return np.array(sino).reshape(sdlist.shape + (nBinsY,nBinsZ,4,))



def mpc_sd_b_proj_4sym(sino, sdlist, nPixels, sBinsY, sBinsZ, cpus=mp.cpu_count()-1):
    phantom = np.zeros(nPixels, dtype=np.float32)  
    dtype = phantom.dtype
    
    phantom_shrd, phantom_shrd_np, l = mpct.create_shared_array(nPixels,lock=True,dtype=dtype)
    
    np.copyto(phantom_shrd_np, phantom)
    
    with mp.Pool(processes=cpus, initializer=mpct.init_shared_array, \
                 initargs=(phantom_shrd.get_obj(), nPixels, dtype, l)) as pool:
        pool.map(mpc_sd_b_proj_s_func4sym, zip(sino,sdlist,\
                                                  itertools.repeat(sBinsY), \
                                                  itertools.repeat(sBinsZ)))

    return phantom_shrd_np



def mpc_sd_f_proj_s_func4sym(args):
    sdlist, sBinsY, sBinsZ = args    
    
    bins0_c = mpct.ctypes_coord(0,sBinsY[0],sBinsZ[0])
    binsN_c = mpct.ctypes_coord(0,sBinsY[1],sBinsZ[1])
    
    nBinsY = sBinsY[1] - sBinsY[0]
    nBinsZ = sBinsZ[1] - sBinsZ[0]
    
    sino_trans = np.zeros(sdlist.shape + (nBinsY, nBinsZ,4), dtype=np.float32)
    sino_trans, sino_trans_c = mpct.ctypes_vars(sino_trans)

    phantom = mpct.shared_to_numpy(mpct.shared_arr['arr'], \
                                   mpct.shared_arr['shape'], \
                                   dtype=mpct.shared_arr['dtype'])

    phantom_c = mpct.ctypes_vars(phantom)[1]
    nX, nY, nZ = mpct.shared_arr['shape']
    dims_c = mpct.ctypes_coord(nX,nY,nZ)    
    nRays = int(np.prod(sdlist.shape))
    
    sdlist_c = (mpct.Ray*nRays)()
    for ray_idx, ray in enumerate(sdlist.flatten()):
        sdlist_c[ray_idx].n = ctypes.c_int(ray[0].size)
 
        if ray != None:
            sdlist_c[ray_idx].X = np.ctypeslib.as_ctypes(ray[0].astype(np.int32))
            sdlist_c[ray_idx].Y = np.ctypeslib.as_ctypes(ray[1].astype(np.int32))
            sdlist_c[ray_idx].Z = np.ctypeslib.as_ctypes(ray[2].astype(np.int32))
            sdlist_c[ray_idx].L = np.ctypeslib.as_ctypes(ray[3].astype(np.float32))
        
    sdlist_c = ctypes.byref(sdlist_c)

    proj_so.forward_proj_rays_sym2(phantom_c, sino_trans_c, sdlist_c,\
                                 dims_c, bins0_c, binsN_c,nRays)

    return sino_trans


def mpc_sd_b_proj_s_func4sym(args):
    sino, sdlist, sBinsY, sBinsZ = args    

    bins0_c = mpct.ctypes_coord(0,sBinsY[0],sBinsZ[0])
    binsN_c = mpct.ctypes_coord(0,sBinsY[1],sBinsZ[1])
        
    sino, sino_c = mpct.ctypes_vars(sino)
        
    phantom = mpct.shared_to_numpy(mpct.shared_arr['arr'], \
                                   mpct.shared_arr['shape'], \
                                   dtype=mpct.shared_arr['dtype'])

    phantom_temp = np.zeros(mpct.shared_arr['shape'], dtype = np.float32)
    phantom_temp, phantom_temp_c = mpct.ctypes_vars(phantom_temp)
    nX, nY, nZ = mpct.shared_arr['shape']
    dims_c = mpct.ctypes_coord(nX,nY,nZ)
    nRays = int(np.prod(sdlist.shape))
    
    sdlist_c = (mpct.Ray*nRays)()
    for ray_idx, ray in enumerate(sdlist.flatten()):
        sdlist_c[ray_idx].n = ctypes.c_int(ray[0].size)
 
        if ray != None:
            sdlist_c[ray_idx].X = np.ctypeslib.as_ctypes(ray[0].astype(np.int32))
            sdlist_c[ray_idx].Y = np.ctypeslib.as_ctypes(ray[1].astype(np.int32))
            sdlist_c[ray_idx].Z = np.ctypeslib.as_ctypes(ray[2].astype(np.int32))
            sdlist_c[ray_idx].L = np.ctypeslib.as_ctypes(ray[3].astype(np.float32))
        
    sdlist_c = ctypes.byref(sdlist_c)
    

    proj_so.back_proj_rays_sym2(phantom_temp_c, sino_c, sdlist_c,\
                         dims_c, bins0_c, binsN_c,nRays)
            
    mpct.shared_arr['lock'].acquire()
    phantom += phantom_temp
    mpct.shared_arr['lock'].release()















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
        the line integra summation of the ray through the phantom
    """
    return np.sum(phantom[(ray[0],ray[1],ray[2])] * ray[3])


def sd_b_proj_ray(phantom, ray, value):
    phantom[(ray[0],ray[1],ray[2])] += ray[3]*value
    

def sd_f_proj(phantom, sdlist):    
    sino = np.zeros(sdlist.shape, dtype=np.float32)
    
    #Iterates through all of the lists    
    for ray_idx, ray in np.ndenumerate(sdlist):
        if ray != None:
            sino[ray_idx] = sd_f_proj_ray(phantom, ray)
    
    return sino


def sd_b_proj(sino, sdlist, nPixels):
    
    phantom = np.zeros(nPixels)
    
    #Iterates through all of the lists    
    for ray_idx, ray in np.ndenumerate(sdlist):
        if ray != None:
            phantom[(ray[0],ray[1],ray[2])] += ray[3]*sino[ray_idx]

    return phantom
















def sphere_inter(Srcs, Trgs, S):

    #Calculates the average intersection length of the sphere and ray
    line = ag.parametric_line(Srcs, Trgs)
    inter_pts = ag.sphere_line_inter(line, S)    
    inter_dist = ag.pts_dist(inter_pts)
    inter_dist[np.isnan(inter_dist)] = 0.0 

    return inter_dist.mean()




"""
DEPRECATED FUNCTIONS
"""

def sd_f_proj_c_ray_var(phantom, sino, X, Y, Z, L, nElem, nX, nY, nZ, sinoElemIdx):
    proj_so.forward_proj_ray_u(phantom, sino, X, Y, Z, L, nElem, \
                               nX, nY, nZ, sinoElemIdx)


def sd_f_proj_c_var(phantom, sdlist):
    sino = np.zeros(sdlist.shape, dtype=np.float32) 
    sino, sino_c = mpct.ctypes_vars(sino)
    
    phantom, phantom_c = mpct.create_ctypes_array(phantom.astype(np.float32))

    count = sd.list_count_elem(sdlist)
    nX, nY, nZ = phantom.shape
    nX = ctypes.c_int(nX)
    nY = ctypes.c_int(nY)
    nZ = ctypes.c_int(nZ)

    for ray_idx, ray in np.ndenumerate(sdlist):
        if ray != None:
            X = mpct.create_ctypes_array(ray[0].astype(np.int32))[1]
            Y = mpct.create_ctypes_array(ray[1].astype(np.int32))[1]
            Z = mpct.create_ctypes_array(ray[2].astype(np.int32))[1]
            L = mpct.create_ctypes_array(ray[3].astype(np.float32))[1]            
   
            nElem = ctypes.c_int(count[ray_idx])
            sinoElemIdx = ctypes.c_int(np.ravel_multi_index(ray_idx,sino.shape))
            
            sd_f_proj_c_ray_var(phantom_c, sino_c, X, Y, Z, L, nElem, nX, nY, nZ, sinoElemIdx)

    return sino


def sd_b_proj_c_ray_var(phantom, sino, X, Y, Z, L, nElem, nX, nY, nZ, sinoElemIdx):
    proj_so.back_proj_ray_u(phantom, sino,X, Y, Z, L, nElem, \
                            nX, nY, nZ, sinoElemIdx)


def sd_b_proj_c_var(sino, sdlist, nPixels):
    sino, sino_c = mpct.ctypes_vars(sino.astype(np.float32))
  
    phantom = np.zeros(nPixels, dtype=np.float32 )
    phantom, phantom_c = mpct.ctypes_vars(phantom)

    count = sd.list_count_elem(sdlist)
    nX, nY, nZ = phantom.shape
    nX = ctypes.c_int(nX)
    nY = ctypes.c_int(nY)
    nZ = ctypes.c_int(nZ)

    for ray_idx, ray in np.ndenumerate(sdlist):
        if ray != None:
            X = mpct.create_ctypes_array(ray[0].astype(np.int32))[1]
            Y = mpct.create_ctypes_array(ray[1].astype(np.int32))[1]
            Z = mpct.create_ctypes_array(ray[2].astype(np.int32))[1]
            L = mpct.create_ctypes_array(ray[3].astype(np.float32))[1]
            
            nElem = ctypes.c_int(count[ray_idx])
            sinoElemIdx = ctypes.c_int(np.ravel_multi_index(ray_idx,sino.shape))
 
            sd_b_proj_c_ray_var(phantom_c, sino_c, X, Y, Z, L, nElem, nX, nY, nZ, sinoElemIdx)

    return phantom

