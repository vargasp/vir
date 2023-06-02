#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:46:48 2023

@author: pvargas21
"""

import numpy as np
import matplotlib.pyplot as plt

import vir.fwhm as fwhm
import vir.psf as psf


nX = 128
nY = 128
nZ = 128

f = 2*np.sqrt(2 * np.log(2))

img1d = psf.gaussian1d(sigma=1000,nX=5001)
print(fwhm.fwhm_1d(img1d)/f )

img1d = psf.gaussian1d(nX=52)
print(fwhm.fwhm_1d(img1d))



center = (0,0)
sigmas = (4,15)
img2d = fwhm.gaussian2d()




print(fwhm.fwhm_orth(img2d))
print(fwhm.fwhm_ang(img2d,90))
print(fwhm.fwhm_ang(img2d,45))
print(fwhm.fwhm_ang(img2d,[45,90]))



center = (8,8,8)
sigmas = (4,5,6)
img3d = fwhm.gaussian3d()

print(fwhm.fwhm_orth(img3d))
print(fwhm.fwhm_ang(img3d,(90,0,0)))
print(fwhm.fwhm_ang(img3d,[(0,0,0),(90,0,0),(45,0,0) ]))

img2d = psf.gaussian2d(mus = (10,0), sigmas=(5,10), nX=129, nY=129, theta =45)
x,y =  fwhm.fwhm_pts(img2d)
plt.imshow(img2d, origin='lower')
plt.plot(y, x)
plt.show()




params = fwhm.fit_error_ellipse(x,y)

xe, ye = fwhm.ellipse_params2xy(params)


