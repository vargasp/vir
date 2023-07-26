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


def moments(x,y, data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.meshgrid(x,y)
    
    
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitGaussian2d(x, y, data):
    data_shape = data.shape

    mu_x0, mu_y0, sigma_x0, sigma_y0, A = moments(data)
    
    
    x = np.ravel(x)
    y = np.ravel(y)
    data = np.ravel(data)

    

p0 = [4,4,1,1,.001]
p = curve_fit(psf.gaussinfunc2d, np.array((x,y)), z,p0=p0)



N = 25
imp2d = psf.gaussian2d(nX=N,nY=N)
imp3d = psf.gaussian3d(nX=N,nY=N,nZ=N)

n = vir.censpace(N)
x2, y2 = np.meshgrid(n,n)
x3, y3, z3 = np.meshgrid(n,n,n)

