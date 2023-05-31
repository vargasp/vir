#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:46:48 2023

@author: pvargas21
"""

import numpy as np
import matplotlib.pyplot as plt

import vir.fwhm as fwhm

nX = 15
nY = 15
nZ = 15


center = (8,8)
sigmas = (4,5)
img2d = fwhm.gaussian2d(1,center, sigmas, nX=nX, nY=nY)

print(fwhm.fwhm_orth(img2d))
print(fwhm.fwhm_ang(img2d,90))
print(fwhm.fwhm_ang(img2d,45))
print(fwhm.fwhm_ang(img2d,[45,90]))



center = (8,8,8)
sigmas = (4,5,6)
img3d = fwhm.gaussian3d(1,center, sigmas, nX=nX, nY=nY,nZ=nZ)

print(fwhm.fwhm_orth(img3d))
print(fwhm.fwhm_ang(img3d,(90,0,0)))
print(fwhm.fwhm_ang(img3d,[(0,0,0),(90,0,0),(45,0,0) ]))
