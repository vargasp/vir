#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:42:09 2023

@author: pvargas21
"""


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy.optimize import curve_fit
from os.path import join, sep, abspath
import vir.affine_transforms as af


def read_file(fname, dirs=None):
    return np.array(Image.open(abspath(join(sep,dirs,fname))))


def reads_gains(proj_dirs, gf_pre='gain.tif', gf_post='gain_post.tif'):
    g1 = read_file(gf_pre, dirs=proj_dirs)
    g2 = read_file(gf_post, dirs=proj_dirs)
    return (g1,g2)


def read_tiff_rows(infiles,rows=None,cols=None):
    """
    Reads a contiguous box of pixels from a series of tiff images and generates
    a stacked array. This can be used to generate sinograms from a set of
    projection images. This method does not read the entire image which can
    improve read speed for large projection image when only a few slice need
    to be reconstructed.

    Parameters
    ----------
    infiles : list or str
        Filename of the tiff image or list of filenames
    rows : int or (2) array like, optional
        row or [row_start, row_end] indices. Default is all rows
    cols : int or (2) array like, optional
        col or [col_start, col_end] indices. Default is all cols

    Returns
    -------
    nd.nparray
    The 

    """


    #If only one file name is provided wrap it in list to make it iterable
    if isinstance(infiles,str):
        infiles = [infiles]
    angles = len(infiles)

    im_obj = Image.open(infiles[0])
    w,h = im_obj.size

    #If row indices are not provided include all rows
    if rows==None:
        rows = np.array([0,h])
    else:
        rows = np.array(rows)

    #If col indices are not provided include all cols
    if cols==None:
        cols = np.array([0,w])
    else:
        cols = np.array(cols)

    #If only 1 index is provided set end to next index
    if rows.size == 1:
        rows = np.append(rows, rows + 1)

    #If only 1 index is provided set end to next index
    if cols.size == 1:
        cols = np.append(cols, cols + 1)

    #Allocate space for the array
    im_arr = np.empty([len(infiles),rows[1]-rows[0],cols[1]-cols[0]],dtype=np.float32)
    
    #Loop though the filenames, extract the subset data, and save it to the array
    for z, infile in enumerate(infiles):
        print(f'\r{(z+1):{int(np.log10(angles))+1}} of {angles} ', end='')
        im_obj = Image.open(infile)        
        im_arr[z,:,:] = np.array(im_obj.crop(box=[cols[0],rows[0],cols[1],rows[1]]))

    print()

    return im_arr


def cog(sino):
    """
    Calculates the center of gravity of a sinogram. The sinogram can be any
    shape, but the bins must be the last axis 

    Parameters
    ----------
    sino : array_like
        Sinogram to calcualte the center of gravity

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.sum(sino*np.arange(1,sino.shape[-1]+1),axis=-1)/sino.sum(axis=-1) - .5


def sine_wave(x, offset, amplitude, phase):
    return np.cos(x + phase) * amplitude + offset

def fit_sine_wave(sino, angs):

    if sino.ndim == 2:
        nAngs, nCols = sino.shape
        nRows = 1
    else:
        nAngs, nRows, nCols = sino.shape 

    cg = cog(sino)
    a = [(nCols-1)/2., (np.max(cg,axis=0) - np.min(cg,axis=0))/2., 0.]                               

    if nRows == 1:
        return curve_fit(sine_wave, angs, cg, p0=a)[0]
    else:
        res = np.empty([nRows,4])
        for row in range(nRows):
            results = curve_fit(sine_wave, angs, cg[:,row], p0=[a[0],a[1][row],a[2]])
            res[row,:3] = results[0]
            res[row,3] = np.diagonal(results[1]).sum()
        
        return res
                

def plot_fit(sino,angs):
    nAngs, nCols = sino.shape 
    cg = cog(sino)
    
    a = [(nCols-1)/2., (np.max(cg) - np.min(cg))/2., 0.]                               

    cf_p, pcov = curve_fit(sine_wave, angs, cg, p0=a)
    
    y = sine_wave(angs, cf_p[0], cf_p[1], cf_p[2])
    print(cf_p, np.diagonal(pcov).sum())
    plt.plot(angs,cg)
    plt.plot(angs,y)
    plt.show()
    
    #plt.plot(angs,y-cg)
    #plt.show()
    


def _org_wave_properites(wp):
    
    idx = wp[2,:] < 0
    wp[2,idx] +=  2*np.pi

    idx = wp[2,:] > 2*np.pi
    wp[2,idx] %= (2*np.pi)
    
    return wp


def org_wave_properites(wp):
    
    wp = _org_wave_properites(wp)
        
    #Deterines how many modes are in the the theta array
    bins = np.linspace(0,np.pi*2,8,endpoint=True)
    hist = np.histogram(wp[2,:],bins=bins)
    nModes = np.sum(hist[0] > 0)
    
    
    if nModes == 2:
        
        idxs = np.argsort(hist[0])[-2:]

        m0_sub = (wp[2,:] > hist[1][idxs[0]]) * (wp[2,:] < hist[1][idxs[0]+1])
        m0_mean = np.mean(wp[2,m0_sub])

        m1_sub = (wp[2,:] > hist[1][idxs[1]]) * (wp[2,:] < hist[1][idxs[1]+1])
        m1_mean = np.mean(wp[2,m1_sub])


        #Aligns thetas accros slices
        if np.isclose(m0_mean-m1_mean, np.pi*2, atol=0.01):
            wp[2,m0_sub] -= np.pi*2
    
        if np.isclose(m1_mean-m0_mean, np.pi*2, atol=0.01):
            wp[2,m1_sub] -= np.pi*2
    
        if np.isclose(m1_mean-m0_mean, np.pi, atol=0.01):
            wp[2,m0_sub] += np.pi
            wp[1,m0_sub] *= -1.0
    
        if np.isclose(m0_mean-m1_mean, np.pi, atol=0.01):
            wp[2,m0_sub] -= np.pi
            wp[1,m0_sub] *= -1.0
    
    slope = np.polyfit(np.arange(wp.shape[1]), wp[1,:], 1)[0]
    if slope < 0:
        wp[2,:] += np.pi
        wp[1,:] *= -1.0
        wp = _org_wave_properites(wp)
    
    
    return wp


def estimate_wobble(sino,angs):
    
    nAngs, nRows, nCols = sino.shape 
    cg = cog(sino)
    
    wave_properties = np.zeros([3,nRows])
    #bounds = ((0,0,0),(nCols,4*nCols,2*np.pi))
    
    for row in range(nRows):
        a = [(nCols-1)/2., (np.max(cg[:,row]) - np.min(cg[:,row]))/2., angs[np.argmax(angs)]]                               

        print(a)

        #cf_p, pcov = curve_fit(sine_wave, angs, cg[:,row], p0=a, bounds=bounds)
        if np.isnan(a[1]):
            cf_p = [0,0,0]
        else:
            cf_p, pcov = curve_fit(sine_wave, angs, cg[:,row], p0=a)
                
        wave_properties[:,row] = cf_p[:3]


    wave_properties = org_wave_properites(wave_properties)
        

    return wave_properties













