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
from scipy.optimize import curve_fit


def moments(imp):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    
    A = imp.max()
    CoMs  = ndimage.center_of_mass(imp)

    profiles = []
    if len(CoMs) == 1:
        profiles.append(imp)
    
    elif len(CoMs) == 2:
        profiles.append(imp[:, int(CoMs[1])])
        profiles.append(imp[int(CoMs[0]), :])

    elif len(CoMs) == 3:
        profiles.append(imp[:, int(CoMs[1]), int(CoMs[2])])
        profiles.append(imp[int(CoMs[0]), :, int(CoMs[2])])
        profiles.append(imp[int(CoMs[0]), int(CoMs[1]), :])

    sigmas = ()
    for i in range(len(CoMs)):
        #Estimate sigma by the root mean squared deviation
        sigmas += (np.sqrt(np.abs((np.arange(profiles[i].size)-CoMs[i])**2*profiles[i]).sum()/profiles[i].sum()),)

    return CoMs, sigmas, A


def fitGaussian2d(x, y, imp):
    imp_shape = imp.shape

    (mu_x0, mu_y0), (sigma_x0, sigma_y0), A = moments(imp)
    mu_x0 = x[int(mu_x0)]
    mu_y0 = y[int(mu_y0)]
    
    p0  = [mu_y0, mu_x0, sigma_y0, sigma_x0, A]
    
    
    x2, y2 = np.meshgrid(x,y)
    x2 = np.ravel(x2)
    y2 = np.ravel(y2)
    z = np.ravel(imp)
    p = curve_fit(psf.gaussinfunc2d, np.array((x2,y2)), z, p0=p0)

    return p
    

def fitGaussian3d(x, y, z, imp):
    imp_shape = imp.shape

    (mu_x0, mu_y0, mu_z0), (sigma_x0, sigma_y0,sigma_z0), A = moments(imp)
    mu_x0 = x[int(mu_x0)]
    mu_y0 = y[int(mu_y0)]
    mu_z0 = z[int(mu_z0)]
    
    p0  = [mu_y0, mu_x0, mu_z0, sigma_y0, sigma_x0, sigma_z0, A]
    
    
    x2, y2, z2 = np.meshgrid(x,y,z)
    x2 = np.ravel(x2)
    y2 = np.ravel(y2)
    z2 = np.ravel(z2)
    imp = np.ravel(imp)
    p = curve_fit(psf.gaussinfunc3d, np.array((x2,y2,z2)), imp, p0=p0)

    return p
    


N = 40
n = vir.censpace(N)
x2, y2 = np.meshgrid(n,n)
x3, y3, z3 = np.meshgrid(n,n,n)

imp2d = psf.gaussian2d(nX=N,nY=N, mus=(1,-2), sigmas=(.5,1))
params = fitGaussian2d(n, n, imp2d)


imp3d = psf.gaussian3d(nX=N,nY=N,nZ=N, mus=(1,-2,3),sigmas=(.5,1,1.5))





"""
print(moments(imp2d))




    

p0 = [4,4,1,1,.001]
p = curve_fit(psf.gaussinfunc2d, np.array((x,y)), z,p0=p0)

"""






