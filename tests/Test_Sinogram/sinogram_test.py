#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:01:39 2023

@author: pvargas21
"""


import numpy as np
import matplotlib.pyplot as plt

import vir
import vir.affine_transforms as af
import vir.sinogram as sg
import vt

from skimage.transform import radon
from scipy.ndimage import affine_transform



nX = 128
nY = 128
nZ = 64
f = 1

"""
3d
"""
phantom = np.zeros([nX*f, nY*f])
phantom[56*f:72*f,56*f:72*f] = 1
phantom = np.tile(phantom, (f*nZ,1,1))
phantom = phantom.transpose([1,2,0])
phantom *= np.arange(f*nZ)

nX, nY, nZ = phantom.shape

nAng = 256*f
angs = np.linspace(0,np.pi*2,nAng,endpoint=False)

sino10 = sg.forward_project_wobble(phantom, angs, 0, 10/180*np.pi, center=(nX/2.-.5,nY/2.-.5,0.5))
sino10 = vir.rebin(sino10, [nAng, nZ, nX])

plt.imshow(sino0[0,:,:], origin='lower')
plt.imshow(sino10[0,:,:], origin='lower')


"""
3d Coorection
"""
a = sg.estimate_wobble(sino10,angs)
center = a[0].mean()
m, y_int = np.polyfit(np.arange(64), a[1], 1)
z_center = -y_int/m
phi = np.arctan(m)
theta = a[2,:].mean()
print(center, z_center, phi*np.pi/180,theta*np.pi/180)


phantom = np.zeros([nX*f, nY*f])
phantom[24*f:40*f,56*f:72*f] = 1
phantom = np.tile(phantom, (f*nZ,1,1))
phantom = phantom.transpose([1,2,0])
phantom *= np.arange(f*nZ)

nX, nY, nZ = phantom.shape

sino10 = sg.forward_project_wobble(phantom, angs, 0, 10/180*np.pi, center=(nX/2.-.5,nY/2.-.5,0.5))
sino10 = vir.rebin(sino10, [nAng, nZ, nX])
sino0 = sg.forward_project_wobble(phantom, angs, 0, 0, center=(nX/2.-.5,nY/2.-.5,0.5))
sino0 = vir.rebin(sino0, [nAng, nZ, nX])


a0 = sg.estimate_wobble(sino0,angs)
a10 = sg.estimate_wobble(sino10,angs)

center = a[0].mean()
m, y_int = np.polyfit(np.arange(64), a[1], 1)
z_center = -y_int/m
phi = np.arctan(m)
theta = a[2,:].mean()
print(center, z_center, phi,theta)









sinoC = sg.correct_wobble(sino10, angs/180*np.pi, -10, 0, center=(0.5, nX/2.0-.5))
plt.imshow(sinoC[0,:,:], origin='lower')



sg.plot_fit(sino10[:,60,:],angs/180*np.pi)



vt.CreatePlot(a[0,:]-64, xtitle='Row Index', ytitle='Center Shift (Offset)', \
              title='CoG Parameter Estimation (Center)')


vt.CreatePlot(a[1,:], xtitle='Row Index', ytitle='Wobble Distance (Amplitude)', \
              title='CoG Parameter Estimation (Wobble Angle/Origin)')

vt.CreatePlot(a[2,:], xtitle='Row Index', ytitle='Start Angle (Phase)', \
              title='CoG Parameter Estimation (Wobble Start Angle)')


vt.CreateImage(sino10[:,60,:].T, title=u'Sino 10\N{DEGREE SIGN} Wobble Row 60')

vt.CreateImage(sino10[:,5,:].T, title=u'Sino 10\N{DEGREE SIGN} Wobble Row 5')

vt.CreateImage(sino0[:,60,:].T, title=u'Sino 0\N{DEGREE SIGN} Wobble Row 60')

vt.CreateImage(sino0[:,5,:].T, title=u'Sino 0\N{DEGREE SIGN} Wobble Row 5')

vt.CreateImage(sinoC[:,60,:].T, title=u'Sino 10\N{DEGREE SIGN} Corrected Wobble Row 60')


vt.CreateImage(sinoC[:,60,:].T - sino0[:,60,:].T, title=u'Difference')


"""
3D Warping
"""

R = af.rotateMat((0.,0.,90.), center=(nX/2, nY/2,0))
P = affine3d(phantom, R)
plt.imshow(P[54:74,54:74,32].T - phantom[54:74,54:74,32].T, origin='lower')




plt.imshow(P[:,64,:].T - phantom[:,64,:].astype(np.float32).T, origin='lower')



nAng = 361
angs = np.linspace(0,360,nAng,endpoint=True)
sino0 = np.zeros([nAng,nZ,nX])
sino10 = np.zeros([nAng,nZ,nX])
for i, ang in enumerate(angs):
    R = af.rotateMat((0,0,ang), center=(nX/2, nY/2,0))
    sino0[i,:,:] = affine3d(phantom, R).sum(axis=1).T
    R = af.rotateMat((0,10,ang), center=(nX/2, nY/2,0))
    sino10[i,:,:] = affine3d(phantom, R).sum(axis=1).T
    
    
    
plt.imshow(P, origin='lower')


plt.imshow(P[:,int(nY/2),:].T, origin='lower')




#Wooble
coords = af.coords_array((nAng,1,nX), ones=True)
coords[:,:,1,:] = 32
w_cords = wobble(coords, (nAng/2,nZ/2,nX/2), angs/180*np.pi, 35, 15)
test = af.coords_transform(sino, coords)
plt.imshow(test[:,0,:],origin='lower')


