#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 12:52:34 2021

@author: vargasp
"""

import importlib
import numpy as np
import matplotlib.pyplot as plt
import phantoms as pt
import filtration as ft
import projection as pj
import visualization_tools as vt
import intersection as inter
import vir 
import spectrums as sp
import xcompy as xc
import proj as pj
import sino_tools as st
import reconstruction as rc
import beamhard as bh
#import blur
#from scipy import ndimage


nDets = 512
nViews = 1000
src_iso = 60
src_det = 107.255
fan_angle = np.deg2rad(49.2)
nPixels = 512

sphere = [0,0,10]

gp = vir.Geom(nViews=nViews)
gf = vir.Geom(nViews=nViews,src_iso=src_iso,src_det=src_det,fan_angle=fan_angle)

dp = vir.Detector2d(nDets=nDets,dDet=gf.fov/nDets)

lin_ints_p = inter.AnalyticSinoParSphere(sphere,gp.Views,dp.W)
lin_ints_f = inter.AnalyticSinoFanSphere(sphere,gf.Views,gf.gammas(nDets),gf.src_iso)

spec = sp.Spectrum("spec80",I0=7.28e8)
#spec = sp.Spectrum("Mono70")

attens = xc.mixatten('H2O',spec.energies)
#attens = np.array([1.0])

det_prob = st.detector_prob(spec.energies,'Gd2O2S',4e-2,7.3)


sino_p = np.sum(spec.energies*det_prob*spec.counts*np.exp(-1.*lin_ints_p[:,:,np.newaxis]*attens),axis=2)
sino_f = np.sum(spec.energies*det_prob*spec.counts*np.exp(-1.*lin_ints_f[:,:,np.newaxis]*attens),axis=2)

lin_ints_pe = -1.0 * np.log(sino_p/(spec.energies*det_prob*spec.counts).sum())
lin_ints_fe = -1.0 * np.log(sino_f/(spec.energies*det_prob*spec.counts).sum())

lin_ints_pe_c = bh.beamhard_cor_table(lin_ints_pe.clip(0),spec.energies, spec.energies*det_prob*spec.counts, gp, spec.effective_mu_water())
lin_ints_fe_c = bh.beamhard_cor_table(lin_ints_fe.clip(0),spec.energies, spec.energies*det_prob*spec.counts, gp, spec.effective_mu_water())

lin_ints_pe_fil = ft.filter_sino(lin_ints_pe_c[:,np.newaxis,:], dCol=dp.dW).squeeze()
rec_p = pj.bp(lin_ints_pe_fil, gp.Views,dp,nPixels=nPixels, dPixel=dp.dW)
rec_f = rc.fan_fbp(lin_ints_fe_c, src_iso, src_det, dDet=fan_angle/nDets*src_det, nPixels=(nPixels,nPixels), dPixels=(dp.dW,dp.dW))

recH_p = st.atten2HU(rec_p,spec.effective_mu_water())
recH_f = st.atten2HU(rec_f,spec.effective_mu_water())


print(recH_p[256,256])
print(recH_f[256,256])
print(spec.effective_mu_water())

plt.plot(recH_p[:,256])
plt.plot(recH_f[:,256])





