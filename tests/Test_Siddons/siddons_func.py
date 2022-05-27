#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 21:49:56 2021

@author: vargasp
"""


import numpy as np




def ave_sdlist(sdlist, ravel=False, nPixels=None):
 
    
    i_pix = np.array([], dtype=int)
    i_len = np.array([], dtype=np.float32)

    for ray_idx, ray in np.ndenumerate(sdlist):

        #Concatenates all of the lists in the lets matrix        
        if ray != None:
            if ravel == True:
                i_pix = np.concatenate((i_pix, ray[0]))
                i_len = np.concatenate((i_len, ray[1]))
            else:
                ray2 = ray_ravel(ray,nPixels)
                i_pix = np.concatenate((i_pix, ray2[0]))
                i_len = np.concatenate((i_len, ray2[1]))

    i_pix, idx =  np.unique(i_pix, return_inverse=True)
    i_len /= sdlist.size
        
    i_len2 = np.zeros(i_pix.size, dtype=np.float32)

    for i, ix in enumerate(idx):
        i_len2[ix] += i_len[i]


    if ravel == True:
        return [i_pix, i_len2]
    else:
        return ray_unravel([i_pix, i_len2],nPixels)
    





def average_inter(inter_list,ave_axes=None):
    max_ray = 16384

    inter_list = np.squeeze(inter_list)

    #Creates the new shape based on axis averaging and moves averaged
    #dimensions to the end
    new_shape = list(inter_list.shape)
    if ave_axes != None:
        for axis in ave_axes:
            inter_list = np.moveaxis(inter_list, axis, -1)
            new_shape.pop(axis)

    new_shape = tuple(new_shape)
    nDets = np.prod(inter_list.shape) / np.prod(new_shape)

    #Defines the arrays based on ther list shape and max elements
    inter_pix = np.zeros(new_shape + (max_ray,), dtype = int)
    inter_len = np.zeros(new_shape + (max_ray,), dtype = float)
   
    #Loops through the new arrays filling them
    for ray_idx, ray in np.ndenumerate(inter_pix[...,0]):
        
        if nDets == 1.0:
            if inter_list[ray_idx] != None:
                nElems = inter_list[ray_idx][0].size
                inter_pix[ray_idx][:nElems] = inter_list[ray_idx][0]
                inter_len[ray_idx][:nElems] = inter_list[ray_idx][1]
        else:
            i_pix = np.array([], dtype=int)
            i_len = np.array([], dtype=float)
            #Concatenates all of the lists in the lets matrix        
            for det_idx, det in np.ndenumerate(inter_list[ray_idx]):
                if det != None:
                    i_pix = np.concatenate((i_pix, det[0]))
                    i_len = np.concatenate((i_len, det[1]))

            i_pix, idx =  np.unique(i_pix, return_inverse=True)
            i_len /= nDets

            inter_pix[ray_idx][:i_pix.size] = i_pix
            for i, ix in enumerate(idx):
                inter_len[ray_idx][ix] += i_len[i]

    Elem_max = np.count_nonzero(inter_len,axis=-1).max()
            
    return inter_pix[...,:Elem_max], inter_len[...,:Elem_max]



def average_inter(sdlist,ave_axes):

    #Creates the new shape based on axis averaging and moves averaged
    #dimensions to the end
    shape_new = sdlist.shape
    for axis in ave_axes:
        sdlist = np.moveaxis(sdlist, axis, -1)
        shape_new = shape_new[:,-1]

    nDets = np.prod(sdlist.shape) / np.prod(shape_new)

    #Defines the arrays based on ther list shape and max elements
    sdlist_new = np.empty(shape_new, dtype=np.object)
    
    #Loops through the new arrays filling them
    for ray_idx, ray in np.ndenumerate(sdlist_new):

        i_pix = np.array([], dtype=int)
        i_len = np.array([], dtype=np.float32)
        
        for ray_idx2, ray2 in np.ndenumerate(sdlist[ray_idx]):

            #Concatenates all of the lists in the lets matrix        
            if ray2 != None:
                i_pix = np.concatenate((i_pix, ray2[0]))
                i_len = np.concatenate((i_len, ray2[1]))

        i_pix, idx =  np.unique(i_pix, return_inverse=True)
        i_len /= nDets
        
        i_lpen2 = np.zeros(i_pix.shape, dtype=np.float32)

        for i, ix in enumerate(idx):
            i_lpen2[ix] += i_len[i]


        sdlist_new[ray_idx] = [i_pix, i_lpen2]


    return sdlist_new
