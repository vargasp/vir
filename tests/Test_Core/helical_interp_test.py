#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 07:19:00 2024

@author: pvargas21
"""


import numpy as np
import matplotlib.pyplot as plt


import vir
import vir.affine_transforms as af
import vir.sinogram as sg


#Acquistion geometry
nRows, nCols =(200,200)
nViews = 400
nAngs = 100
z_trans = 50 #Z translation (# of rows/revolution)

hel_geom = vir.Geom(nViews, coverage=nViews/nAngs*2*np.pi,zTran=z_trans)


intZ = [6,6,7,8,6,7,8,]
idxV, idxL, idxU, dL, dU = hel_geom.interpZ(intZ,5,nRows,all_views=True)



