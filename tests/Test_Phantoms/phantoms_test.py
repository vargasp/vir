#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:22:55 2020

@author: vargasp
"""

"""
%matplotlib qt5
%matplotlib inline
"""

import importlib
import numpy as np
import matplotlib.pyplot as plt
import phantoms as pt
import vir 
import visualization_tools as vt


dMinSphere = 0.4 #[Micronns]
SphereScale = dMinSphere/0.8 #[Unitless]
dDet = 0.5 #[Micronns]
nDets = (32/dDet)


importlib.reload(pt)

derenzo_spheres = pt.DerenzoPhantomSpheres(scale=SphereScale)
pd = pt.Phantom(spheres = derenzo_spheres)


pd_d = pt.DiscretePhantom(nPixels=128*4,dPixel=.25/4)

pd_d.updatePhantomDiscrete(pd.S)

plt.imshow(pd_d.phantom[:,:,64])



for i in range(128*4):
    trail = str(i).zfill(4)
    vt.CreateTiffImage(pd_d.phantom[:,:,i], outfile = 'Test/derenzo' + trail)
