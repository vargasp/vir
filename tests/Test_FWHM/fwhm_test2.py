# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:27:59 2023

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt

import vir.psf as psf
import vir.fwhm as fwhm


imp2d = psf.gaussian2d(nX=25,nY=25)
imp3d = psf.gaussian3d(nX=25,nY=25,nZ=25)


cX, cY= np.unravel_index(np.argmax(imp2d), imp2d.shape)
p0 = np.tile([cX,cY],[2,1])
p1 = np.array([[cX,24],[24,24]])

profiles2s = fwhm.profile_img(imp2d, p0, p1, dPix=1.0)
profiles2 = fwhm.impulse_profile(imp2d,samples=40)
fwhms2 = fwhm.fwhm_edge_profiles(profiles2, v=0.5)




cX, cY, cZ = np.unravel_index(np.argmax(imp3d), imp3d.shape)
p0 = np.tile([cX,cY,cZ],[2,1])
p1 = np.array([[cX,24,cZ],[24,24,cZ]])

profiles3s = fwhm.profile_img(imp3d, p0, p1, dPix=1.0)
profiles3 = fwhm.impulse_profile(imp3d,samples=40)
fwhms = fwhm.fwhm_edge_profiles(profiles3, v=0.5)
imp_chars = fwhm.impulse_characteristics(imp3d)