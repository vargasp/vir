#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:59:05 2023

@author: pvargas21
"""




import numpy as np
import matplotlib.pyplot as plt

import vir
import vir.psf as psf
import vir.fwhm as fwhm

from scipy import ndimage


def moments(imp):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    
    A = imp.max()
    x, y  = ndimage.center_of_mass(imp)

    profile_x = imp[:, int(y)]
    profile_y = imp[int(x), :]

    #Estimate sigma by the root mean squared deviation
    sigma_x = np.sqrt(np.abs((np.arange(profile_x.size)-x)**2*profile_x).sum()/profile_x.sum())
    sigma_y = np.sqrt(np.abs((np.arange(profile_y.size)-y)**2*profile_y).sum()/profile_y.sum())

    return x, y, sigma_x, sigma_y, A


def fitGaussian2d(x, y, data):
    data_shape = data.shape

    mu_x0, mu_y0, sigma_x0, sigma_y0, A = moments(data)
    
    
    x = np.ravel(x)
    y = np.ravel(y)
    data = np.ravel(data)
    p = curve_fit(psf.gaussinfunc2d, np.array((x,y)), z,p0=p0)
    


N = 15
n = vir.censpace(N)
x2, y2 = np.meshgrid(n,n)
x3, y3, z3 = np.meshgrid(n,n,n)

imp2d = psf.gaussian2d(nX=N,nY=N, mus=(5,-2), sigmas=(.5,1))
print(moments(imp2d))


imp3d = psf.gaussian3d(nX=N,nY=N,nZ=N)

    

p0 = [4,4,1,1,.001]
p = curve_fit(psf.gaussinfunc2d, np.array((x,y)), z,p0=p0)








