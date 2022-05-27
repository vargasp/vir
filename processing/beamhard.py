#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 19:04:28 2020

@author: vargasp
"""
import numpy as np

import vir
import intersection as inter
import projection as pj
import xcompy as xc

from scipy import interpolate


def beamhard_cor_table(line_ints, energies, counts, g, atten_eff):
    '''
    Computes the beam hardening correction coeffs based on Hsieh (section 7.6.2)
    based on a circular water phantom

    spectrum: 2d array of the incident xray spectrum [energies (KeV), relative intensity]
    I0: spectrum phonton count per channel
    FocalLength: Distance from isocenter to source
    src_det: Distance from the source to the detector
    '''

    attens = xc.mixatten('H2O',energies)
    
    path_lengths = np.linspace(0,80,1000)
    path_lengths_p = -1.0*np.log(np.sum(counts*np.exp(-1.0*np.outer(path_lengths,attens)),axis=1)/np.sum(counts))/atten_eff

    f = interpolate.interp1d(path_lengths_p*atten_eff, path_lengths*atten_eff)

    return f(line_ints.clip(0))


def beamhard_cor(line_ints, coefs,mu):
    '''
    Corrects for beamhardening (water only)
    
    line_ints: logged sinogram data
    coefs: array of beam hardening coefficients
    '''
    line_ints = line_ints/mu
    
    return mu*(coefs[0]*line_ints**2+coefs[1]*line_ints + coefs[2])


def beamhard_coefs2(energies, counts, g, atten_eff):
    '''
    Computes the beam hardening correction coeffs based on Hsieh (section 7.6.2)
    based on a circular water phantom

    spectrum: 2d array of the incident xray spectrum [energies (KeV), relative intensity]
    I0: spectrum phonton count per channel
    FocalLength: Distance from isocenter to source
    src_det: Distance from the source to the detector
    '''
   
    nBins = 512
    sphere = [0,0,20]
    Bins = vir.censpace(nBins,d=0.09756581068603154)
    attens = xc.mixatten('H2O',energies)
    
    if g.geom == 'Fan beam':
        path_lengths = inter.AnalyticSinoFanSphere(sphere,0,g.gammas(nBins),g.src_iso)
        path_lengths = np.squeeze(path_lengths)[0:int(nBins/2)]
    elif g.geom == 'Parallel beam':
        path_lengths = inter.AnalyticSinoParSphere(sphere,0,Bins)
        path_lengths = np.squeeze(path_lengths)[0:int(nBins/2)]
        

    path_lengths_p = -1.0*np.log(np.sum(counts*np.exp(-1.0*np.outer(path_lengths,attens)),axis=1)/np.sum(counts))/atten_eff


    return np.polyfit(path_lengths_p,path_lengths,2)
    


def beamhard_coefs(energies, counts, det_prob=1.0, FocalLength=60,src_det=107.255,alpha=0):
    '''
    Computes the beam hardening correction coeffs based on Hsieh (section 7.6.2)
    based on a circular water phantom

    spectrum: 2d array of the incident xray spectrum [energies (KeV), relative intensity]
    I0: spectrum phonton count per channel
    FocalLength: Distance from isocenter to source
    src_det: Distance from the source to the detector
    '''

    
    if alpha == 0:
        energy_weights = det_prob
    else:
        energy_weights = alpha*energies*det_prob
    
    nViews = 1
    nCols = 896
    I0 = np.trapz(counts*energy_weights,x=energies)
    
    attens = xc.mixatten('H2O',energies)
    energy_eff = np.average(energies, weights=counts)
    atten_eff = xc.mixatten('H2O', [energy_eff])[0]

    phantom = water_tub_phantom()
    path_lengths = pj.fan_forwardproject(phantom, FocalLength, src_det, nViews, nCols, dPixel=.1, dDet=.1)
    path_lengths = np.squeeze(path_lengths)[0:int(nCols/2)]

    path_lengths_p = -1.0*np.log(np.trapz(counts*np.exp(-1.0*np.outer(path_lengths,attens))*energy_weights, x=energies,axis=1)/I0)/atten_eff

    return np.polyfit(path_lengths_p,path_lengths,2)
    

def water_tub_phantom(npixels=512,dpixel=.1,tub_radius=20):
    #npixels: dimension of phantom [pixels]
    #dpixel: pixel size [arbitrary length]
    #tub_radius: length or tub of water's radius [arbirart length]
    
    x,y = np.indices((npixels,npixels))
    return (x - npixels/2)**2 + (y - npixels/2)**2 < (tub_radius/dpixel)**2

    



