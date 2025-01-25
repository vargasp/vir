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

z_vals = np.zeros(16)
z_vals[4:12]=np.arange(1,9)
phantom *= np.tile(z_vals,4*f)

nX, nY, nZ = phantom.shape

nAng = 256*f
angs = np.linspace(0,np.pi*2,nAng,endpoint=False)

sino0 = sg.forward_project_wobble(phantom, angs, 0, 0, center=(nX/2.-.5,nY/2.-.5,0.5))
sino0 = vir.rebin(sino0, [nAng, nZ, nX])
sino10 = sg.forward_project_wobble(phantom, angs, 0, 20/180*np.pi, center=(nX/2.-.5,nY/2.-.5,0.5))
sino10 = vir.rebin(sino10, [nAng, nZ, nX])

plt.imshow(sino0[128,:,:], origin='lower')
plt.show()
plt.imshow(sino10[128,:,:], origin='lower')
plt.show()


"""
3d Correction
"""
a = sg.estimate_wobble(sino10,angs)
center = a[0].mean()
m, y_int = np.polyfit(np.arange(64), a[1], 1)
z_center = -y_int/m
phi = np.arctan(m)
theta = a[2,:].mean()
print(f'Center: {center:.2f}, Z Center: {z_center:.2f}, Phi: {phi/np.pi*180:.2f}, Theta:{theta*np.pi/180:2f}')



nX = 128
nY = 128
nZ = 160
f = 1

"""
3d Center
"""
phantomC = np.zeros([nX*f, nY*f])
phantomC[56*f:72*f,56*f:72*f] = 1
phantomC = np.tile(phantomC, (f*nZ,1,1))
phantomC = phantomC.transpose([1,2,0])
phantomC *= np.arange(f*nZ)

nX, nY, nZ = phantomO.shape

nAng = 256*f
angs = np.linspace(0,np.pi*2,nAng,endpoint=False)

sinoC0 = sg.forward_project_wobble(phantomC, angs, 0, 0, center=(nX/2.-.5,nY/2.-.5,nZ/2.-.5))
sinoC0 = vir.rebin(sinoC0, [nAng, nZ, nX])
sinoC0 = sinoC0[:,16:144,:]
sinoC10 = sg.forward_project_wobble(phantomC, angs, 0, 10/180*np.pi, center=(nX/2.-.5,nY/2.-.5,nZ/2.-.5))
sinoC10 = vir.rebin(sinoC10, [nAng, nZ, nX])
sinoC10 = sinoC10[:,16:144,:]


plt.imshow(sinoC0[0,:,:], origin='lower')
plt.imshow(sinoC10[0,:,:], origin='lower')

C0 = sg.estimate_wobble(sinoC0,angs)
centerC0 = C0[0].mean()
mC0, y_intC0 = np.polyfit(np.arange(128), C0[1], 1)
z_centerC0 = -y_intC0/mC0
phiC0 = np.arctan(mC0)
thetaC0 = C0[2,:].mean()
print(f'Center: {centerC0:.2f}, Z Center: {z_centerC0:.2f}, Phi: {phiC0/np.pi*180:.2f}, Theta:{thetaC0*np.pi/180:2f}')


C10 = sg.estimate_wobble(sinoC10,angs)
centerC10 = C10[0].mean()
mC10, y_intC10 = np.polyfit(np.arange(128), C10[1], 1)
z_centerC10 = -y_intC10/mC10
phiC10 = np.arctan(mC10)
thetaC10 = C10[2,:].mean()
print(f'Center: {centerC10:.2f}, Z Center: {z_centerC10:.2f}, Phi: {phiC10/np.pi*180:.2f}, Theta:{thetaC10*np.pi/180:2f}')


"""
3d Offset
"""



phantomO = np.zeros([nX*f, nY*f])
phantomO[24*f:40*f,56*f:72*f] = 1
phantomO = np.tile(phantomO, (f*nZ,1,1))
phantomO = phantomO.transpose([1,2,0])
phantomO *= np.arange(f*nZ)

nX, nY, nZ = phantomO.shape

sinoO0 = sg.forward_project_wobble(phantomO, angs, 0, 0, center=(nX/2.-.5,nY/2.-.5,nZ/2.-.5))
sinoO0 = vir.rebin(sinoO0, [nAng, nZ, nX])
sinoO0 = sinoO0[:,16:144,:]
sinoO10 = sg.forward_project_wobble(phantomO, angs, 0, 10/180*np.pi, center=(nX/2.-.5,nY/2.-.5,nZ/2.-.5))
sinoO10 = vir.rebin(sinoO10, [nAng, nZ, nX])
sinoO10 = sinoO10[:,16:144,:]


plt.imshow(sinoO0[0,:,:], origin='lower')
plt.imshow(sinoO10[0,:,:], origin='lower')

O0 = sg.estimate_wobble(sinoO0,angs)
centerO0 = O0[0].mean()
mO0, y_intO0 = np.polyfit(np.arange(128), O0[1], 1)
z_centerO0 = -y_intO0/mO0
phiO0 = np.arctan(mO0)
thetaO0 = O0[2,:].mean()
print(f'Center: {centerO0:.2f}, Z Center: {z_centerO0:.2f}, Phi: {phiO0/np.pi*180:.2f}, Theta:{thetaO0*np.pi/180:2f}')


O10 = sg.estimate_wobble(sinoO10,angs)
centerO10 = O10[0].mean()
mO10, y_intO10 = np.polyfit(np.arange(128), O10[1], 1)
z_centerO10 = -y_intO10/mO10
phiO10 = np.arctan(mO10)
thetaO10 = O10[2,:].mean()
print(f'Center: {centerO10:.2f}, Z Center: {z_centerO10:.2f}, Phi: {phiO10/np.pi*180:.2f}, Theta:{thetaO10*np.pi/180:2f}')


labels = ['Center 0', 'Center 10', 'Offset 0', 'Offset 10']
p1 = np.stack([C0[1],C10[1],O0[1],O10[1]]).T
vt.CreatePlot(p1, labels=labels)

p2 = np.stack([C0[2],C10[2],O0[2],O10[2]]).T
vt.CreatePlot(p2, labels=labels)



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


