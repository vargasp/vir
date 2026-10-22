#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 15:35:35 2021

@author: vargasp
"""
import numpy as np
from skimage.transform import radon, iradon

import pyrm_funcs as pyf


def SIRT_py(nPixels,sino,sdlist,d,dPixel,iters=10,gamma=1.0,\
            W_sino=None,W_img=None,initial=None,save_file=False):
    
    image_iters = np.zeros(nPixels + (iters,))
    
    if W_sino is None:
        W_sino =  pyf.py_forward_proj2(np.ones(nPixels),sdlist,d,dPixel)

    if W_img is None:
        W_img = pyf.py_back_proj2(np.where(W_sino>0,1.0,0),sdlist,d,nPixels,dPixel)

    if initial is None:
        image_iters[...,0] = np.zeros(nPixels)
    else:
        image_iters[...,0] = initial


    idx1 = np.nonzero(W_sino)
    idx2 = np.nonzero(W_img)

    for i in range(1,iters):
        print("SIRT Iteration:",i)
        sino_est = pyf.py_forward_proj2(image_iters[...,i-1], sdlist, d,dPixel)
        sino_est[idx1] = (sino[idx1] - sino_est[idx1])/W_sino[idx1]
        image_update = pyf.py_back_proj2(sino_est, sdlist, d, nPixels,dPixel)
        image_update[idx2] /= W_img[idx2]
        image_iters[...,i] = image_iters[...,i-1] + gamma*image_update
        if save_file:
            np.save("i_"+str(i)+"_"+str(save_file)+'_'.join([str(n) for n in nPixels]), image_iters[...,i])

    return image_iters


def SIRT_weights(nPixels, inter_pix, inter_len):
    W_img = np.zeros(np.prod(nPixels))

    for p_idx, pix in np.ndenumerate(inter_pix):
        W_img[pix] += inter_len[p_idx]
    
    W_img = np.reshape(W_img, nPixels)
    W_sino = np.sum(inter_len,axis = -1)
    
    return (W_sino, W_img)
    

def forward_proj1(phantom, inter_pix, inter_len):
    sino = np.zeros(inter_pix.shape[:-1])
    
    #Iterates through all of the lists    
    for ray_idx, ray in np.ndenumerate(sino):
        if inter_len[ray_idx][0] != 0.0:
            idx = np.unravel_index(inter_pix[ray_idx], phantom.shape)
            sino[ray_idx] = np.sum(phantom[idx] * inter_len[ray_idx])
    
    return sino


def forward_proj2(phantom, inter_pix, inter_len):
    sino = np.zeros(inter_pix.shape[:-2])
    
    #Iterates through all of the lists    
    for ray_idx, ray in np.ndenumerate(sino):
        if inter_len[ray_idx][0] != 0.0:
            idx = inter_pix[ray_idx]
            sino[ray_idx] = np.sum(phantom[idx] * inter_len[ray_idx])
    
    return sino

"""
def forward_proj2(phantom, inter_pix, inter_len):
    sino = np.zeros(inter_pix.shape[:-2])
    
    #Iterates through all of the lists    
    for ray_idx, ray in np.ndenumerate(sino):
        if inter_len[ray_idx][0] != 0.0:
            sino[ray_idx] = np.sum(phantom[idx] * inter_len[ray_idx])
    
    return sino
"""


def back_proj2(sino, inter_pix, inter_len, nPixels):
    image = np.zeros(nPixels)
    
    #Iterates through all of the lists    
    for ray_idx, ray in np.ndenumerate(sino):
        if sino[ray_idx] != 0:
            idx = np.unravel_index(inter_pix[ray_idx], image.shape)
        
            image[idx] += inter_len[ray_idx]*sino[ray_idx]
    
    return image



def SIRT1(nPixels, sino, inter_pix, inter_len, iters=10, gamma=1.0):
    W_sino, W_img = SIRT_weights(nPixels, inter_pix, inter_len)

    image_iters = np.zeros(nPixels + (iters,))
    image_iters[...,0] = np.zeros(nPixels)

    idx = np.nonzero(W_sino)

    for i in range(1,iters):
        print(i)
        sino_est = forward_proj2(image_iters[...,i-1], inter_pix, inter_len)
        sino_est[idx] = (sino[idx] - sino_est[idx])/W_sino[idx]
        image_update = back_proj2(sino_est, inter_pix, inter_len, nPixels)/W_img
        image_iters[...,i] = image_iters[...,i-1] + gamma*image_update

    return image_iters

def SIRT2(nPixels, sino, inter_pix, inter_len, iters=10, gamma=1.0):
    W_sino, W_img = SIRT_weights(nPixels, inter_pix, inter_len)

    inter_pix2 = np.stack(np.unravel_index(inter_pix, nPixels), axis=-1)

    image_iters = np.zeros(nPixels + (iters,))
    image_iters[...,0] = np.zeros(nPixels)

    idx = np.nonzero(W_sino)

    for i in range(1,iters):
        print(i)
        sino_est = forward_proj2(image_iters[...,i-1], inter_pix2, inter_len)
        sino_est[idx] = (sino[idx] - sino_est[idx])/W_sino[idx]
        image_update = back_proj2(sino_est, inter_pix2, inter_len, nPixels)/W_img
        image_iters[...,i] = image_iters[...,i-1] + gamma*image_update

    return image_iters






def SIRT_py3d_r(nPixels, sino, inter_pix, inter_len, iters=10, gamma=1.0):
    W_sino, W_img = SIRT_weights(nPixels, inter_pix, inter_len)
    W_img = W_img +  np.rot90(W_img,1,axes=(0,1))
    
    
    image_iters = np.zeros(nPixels + (iters,))
    image_iters[:,:,:,0] = np.zeros(nPixels)

    idx = np.nonzero(W_sino)

    for i in range(1,iters):
        print(i)
        sino_est = forward_proj2(image_iters[:,:,:,i-1], inter_pix, inter_len)
        sino_est[idx] = (sino[idx] - sino_est[idx])/W_sino[idx]
        image_update = back_proj2(sino_est, inter_pix, inter_len, nPixels)
        image_update = (image_update +  np.rot90(image_update,1,axes=(0,1)))/W_img
        
        image_iters[:,:,:,i] = image_iters[:,:,:,i-1] + gamma*image_update

    return image_iters






def SIRT_par2d(nPixels, sino, thetas, W_sino, W_img, iters=10, gamma=1.0):


    image_iters = np.zeros(nPixels + (iters,))
    image_iters[:,:,0] = np.zeros(nPixels)

    for i in range(1,iters):
        sino_est = radon(image_iters[:,:,i-1], theta=np.rad2deg(thetas))
        image_update = iradon((sino - sino_est)/W_sino, theta=np.rad2deg(thetas), filter_name=None)/W_img
        image_iters[:,:,i] = image_iters[:,:,i-1] + gamma*image_update

    return image_iters









def forward_proj_old(phantom, inter_pix, inter_len, DetYs, DetZs, dPixel):

    nPixels = phantom.shape
    nTheta,nPhi,nElems = inter_pix.shape

    sino = np.zeros([DetYs.size,DetZs.size,nTheta,nPhi])
    
    #Loops through the anglges
    for j in range(nTheta): 
        for k in range(nPhi):
            idx_l = np.squeeze(np.argwhere(inter_len[j,k,:]))
  
            if idx_l.size > 0:
                idx_p = np.unravel_index(inter_pix[j,k,idx_l],nPixels)
                idx_x = np.atleast_1d(idx_p[0])           
                idx_ys = np.add.outer(np.floor(DetYs/dPixel), np.atleast_1d(idx_p[1])).astype(int)
                idx_zs = np.add.outer(np.floor(DetZs/dPixel), np.atleast_1d(idx_p[2])).astype(int)        

                if idx_ys.ndim == 1:
                    idx_ys = idx_ys[np.newaxis,:]

                if idx_zs.ndim == 1:
                    idx_zs = idx_zs[np.newaxis,:]
              
                for y, idx_y in enumerate(idx_ys):
                    idx_ly = np.nonzero((idx_y >= 0) & (idx_y < nPixels[1]))

                    for z, idx_z in enumerate(idx_zs):
                        idx_lz = np.nonzero((idx_z >= 0) & (idx_z < nPixels[2]))

                        idx_l = np.intersect1d(idx_ly, idx_lz)
                        idx_pn = (idx_x[idx_l],idx_y[idx_l],idx_z[idx_l])

                        sino[y,z,j,k] = np.sum(phantom[idx_pn] * inter_len[j,k,idx_l])
             
    return sino


def back_proj_old(sino, inter_pix, inter_len, DetYs, DetZs, nPixels, dPixel):

    nTheta,nPhi,nElems = inter_pix.shape

    phantom = np.zeros(nPixels)

    #Loops through the anglges
    for j in range(nTheta): 
        for k in range(nPhi):
            idx_l = np.squeeze(np.argwhere(inter_len[j,k,:]))
  
            if idx_l.size > 0:
                idx_p = np.unravel_index(inter_pix[j,k,idx_l], nPixels)
                idx_x = np.atleast_1d(idx_p[0])
                idx_ys = np.add.outer(np.floor(DetYs/dPixel), np.atleast_1d(idx_p[1])).astype(int)
                idx_zs = np.add.outer(np.floor(DetZs/dPixel), np.atleast_1d(idx_p[2])).astype(int)        

                if idx_ys.ndim == 1:
                    idx_ys = idx_ys[np.newaxis,:]

                if idx_zs.ndim == 1:
                    idx_zs = idx_zs[np.newaxis,:]

                for y, idx_y in enumerate(idx_ys):
                    idx_ly = np.nonzero((idx_y >= 0) & (idx_y < nPixels[1]))

                    for z, idx_z in enumerate(idx_zs):
                        idx_lz = np.nonzero((idx_z >= 0) & (idx_z < nPixels[2]))

                        idx_l = np.intersect1d(idx_ly, idx_lz)
                        idx_pn = (idx_x[idx_l],idx_y[idx_l],idx_z[idx_l])

                        if idx_l.size > 0:                     
                            phantom[idx_pn] += sino[y,z,j,k] * inter_len[j,k,idx_l]
 
    return phantom


def SIRT_old(sino, inter_pix, inter_len, DetYs, DetZs, nPixels,dPixel, iters=5, gamma=1):
    
    image = np.zeros(nPixels)

    for i in range(iters):
        sino_est = forward_proj_old(image, inter_pix, inter_len, DetYs, DetZs, dPixel)
        sino - sino_est
        
        image =  back_proj_old(sino, inter_pix, inter_len, DetYs, DetZs, nPixels, dPixel)
        image += gamma

    return image
