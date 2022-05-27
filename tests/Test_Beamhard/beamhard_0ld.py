import numpy as np
import projection as pj
import xcompy as xc


def beamhard_cor(line_ints, coefs,mu):
    '''
    Corrects for beamhardening (water only)
    
    line_ints: logged sinogram data
    coefs: array of beam hardening coefficients
    '''
    line_ints = line_ints/mu
    
    return mu*(coefs[0]*line_ints**2+coefs[1]*line_ints + coefs[2])


def beamhard_coefs(energies, counts, det_prob, FocalLength=60,src_det=107.255,alpha=0):
    '''
    Computes the beam hardening correction coeffs based on Hsieh (section 7.6.2)
    based on a circular water phantom

    spectrum: 2d array of the incident xray spectrum [energies (KeV), relative intensity]
    I0: spectrum phonton count per channel
    FocalLength: Distance from isocenter to source
    src_det: Distance from the source to the detector
    '''

    
    if alpha == 0:
        energy_weights = np.ones(energies.size)*det_prob
    else:
        energy_weights = alpha*energies*det_prob
    
    nViews = 1
    nCols = 896
    I0 = np.sum(counts*energy_weights)
    
    attens = xc.mixatten('H2O',energies)
    energy_eff = np.average(energies, weights=counts)
    atten_eff = xc.mixatten('H2O', [energy_eff])[0]

    phantom = water_tub_phantom()
    path_lengths = pj.fan_forwardproject(phantom, FocalLength, src_det, nViews, nCols, dPixel=.1, dDet=.1)
    path_lengths = np.squeeze(path_lengths)[0:int(nCols/2)]

    sino_p = np.zeros(path_lengths.shape)
    for i, intensity in enumerate(counts):
        sino_p += intensity*np.exp(-1.0*attens[i]*path_lengths)*energy_weights[i]

    path_lengths_p = -1.0*np.log(sino_p/I0)/atten_eff

    return np.polyfit(path_lengths_p,path_lengths,2)
    

def water_tub_phantom(npixels=512,dpixel=.1,tub_radius=20):
    #npixels: dimension of phantom [pixels]
    #dpixel: pixel size [arbitrary length]
    #tub_radius: length or tub of water's radius [arbirart length]
    
    x,y = np.indices((npixels,npixels))
    return (x - npixels/2)**2 + (y - npixels/2)**2 < (tub_radius/dpixel)**2

