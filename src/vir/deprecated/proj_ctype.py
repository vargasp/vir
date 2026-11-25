# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 17:54:27 2025

@author: varga
"""

import ctypes

import numpy as np
import vir.mpct as mpct


try:
    proj_so = ctypes.CDLL(__file__.rsplit("/",1)[0] + "/projection.so")
    #print("Linux/Mac C code compiled and loaded")
 
except:
    try:
        proj_so = ctypes.CDLL(__file__.rsplit("\\",1)[0] + "\\projection.dll")
        #print("PC C code compiled and loaded")
    except:
        print("Warning. Projection shared object libray not present.")



def sd_f_proj_c(phantom, sino, sdlist_c, ravel=True, C=False,dims=None,nRays=None):

    if C:
        sino_c = sino
        phantom_c = phantom
        nX, nY, nZ = dims
        dims_c = mpct.ctypes_coord(nX, nY, nZ)
        nRays_c = mpct.c_int(nRays)
    else:
        sino, sino_c = mpct.ctypes_vars(sino)
        phantom, phantom_c = mpct.ctypes_vars(phantom)
        nX, nY, nZ = phantom.shape
        dims_c = mpct.ctypes_coord(nX, nY, nZ)
        nRays_c = mpct.c_int(sino.size)

    if ravel:
        proj_so.forward_proj_ravel_rays_struct(phantom_c, sino_c, sdlist_c, \
                                               nRays_c)
    else:
        proj_so.forward_proj_unravel_rays_struct(phantom_c, sino_c, sdlist_c, \
                               dims_c, nRays_c)

    """
    for ray_idx, ray_c in np.ndenumerate(sdlist_c):
        if ray_c != None:   
            sinoElemIdx = mpct.c_int(np.ravel_multi_index(ray_idx,sino.shape))
            proj_so.forward_proj_ray_u_struct(phantom_c, sino_c, ray_c, \
                                       dims_c, sinoElemIdx)
    """

def sd_b_proj_c(phantom, sino, sdlist_c, ravel=True, C=False,dims=None,nRays=None):

    if C:
        sino_c = sino
        phantom_c = phantom
        nX, nY, nZ = dims
        dims_c = mpct.ctypes_coord(nX, nY, nZ)
        nRays_c = mpct.c_int(nRays)
    else:
        sino, sino_c = mpct.ctypes_vars(sino)
        phantom, phantom_c = mpct.ctypes_vars(phantom)
        nX, nY, nZ = phantom.shape
        dims_c = mpct.ctypes_coord(nX, nY, nZ)
        nRays_c = mpct.c_int(sino.size)

    if ravel:
        proj_so.back_proj_ravel_rays_struct(phantom_c, sino_c, sdlist_c, \
                               nRays_c)
    else:
        proj_so.back_proj_unravel_rays_struct(phantom_c, sino_c, sdlist_c, \
                               dims_c, nRays_c)
    

    """
    if flat:
        proj_so.back_proj_rays_u_struct(phantom_c, sino_c, sdlist_c, \
                                   dims_c, mpct.c_int(sino.size))
    else:
        for ray_idx, ray_c in np.ndenumerate(sdlist_c):
            if ray_c != None:
                sinoElemIdx = mpct.c_int(np.ravel_multi_index(ray_idx,sino.shape))
                proj_so.back_proj_ray_u_struct(phantom_c, sino_c, ray_c, \
                            dims_c, sinoElemIdx)
  """

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

