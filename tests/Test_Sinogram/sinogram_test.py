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


def phantom1(f):
    nX,nY,nZ = (128,128,64)

    phantom = np.zeros([nX*f, nY*f, nZ*f])
    phantom[56*f:72*f,56*f:72*f,8*f:56*f] = 1
    
    return phantom


def phantom3(f):
    nX,nY,nZ = (128,128,64)

    phantom = np.zeros([nX*f, nY*f])
    phantom[56*f:72*f,56*f:72*f] = 1
    phantom = np.tile(phantom, (f*nZ,1,1))
    phantom = phantom.transpose([1,2,0])

    z_vals = np.zeros(16)
    z_vals[4:12]=np.arange(1,9)
    phantom *= np.tile(z_vals,4*f)
    
    return phantom


def calib_det_orient_TM(r, x, center):
    T = af.transMat((0,0,x))
    R = af.rotateMat((r,0,0), center=center)

    return T, R


def calib_det_orient(sino3d, Angs, ang, r, x):
    nAngs, nRows, nCols = sino3d.shape
    
    coords = af.coords_array((1,nRows,nCols), ones=True)
    coords[:,:,0,:] = np.interp(ang,Angs,np.arange(nViews))
    
    center = np.array([1.0,nRows,nCols])/2.0 - 0.5
    T, R = calib_det_orient_TM(-r, -x, center)
    
    TRC = np.linalg.inv(T @ R) @ coords
    return np.squeeze(af.coords_transform(sino3d, TRC))


def calib_precesion_TM(ang, phi, theta, center):
    r = phi*np.cos(ang + theta)
    z = np.sin(np.pi/2 - phi)
    h_xy = np.cos(np.pi/2 - phi) 
    
    x = np.cos(ang + theta)*h_xy
    s = np.sqrt(x**2 + z**2)

    S = af.scaleMat((1,1/s,1))
    R = af.rotateMat((r,0,0), center=center)
    
    return S, R


def calib_precesion(sino3d, Angs, ang, phi, theta, center):
    nViews, nRows, nCols = sino3d.shape

    coords = af.coords_array((1,nRows,nCols), ones=True)
    coords[:,:,0,:] = np.interp(ang,Angs,np.arange(nViews))

    S, R = calib_precesion_TM(ang, phi, theta, center)

    SRC = np.linalg.inv(S @ R) @ coords
    return np.squeeze(af.coords_transform(sino3d, SRC))


def calib_both(sino3d, Angs, ang, phi, theta, center, rd, xd):
    nAngs, nRows, nCols = sino3d.shape
    
    coords = af.coords_array((1,nRows,nCols), ones=True)
    coords[:,:,0,:] = np.interp(ang,Angs,np.arange(nViews))

    #Detector Transforms
    center_det = np.array([1.0,nRows,nCols])/2.0 - 0.5
    Td, Rd = calib_det_orient_TM(-rd, -xd, center_det)
    
    #Precession Transforms
    Sp, Rp = calib_precesion_TM(ang, phi, theta, center)
    
    SRTR = np.linalg.inv(Sp @ Rp @ Td @ Rd) @ coords
    return np.squeeze(af.coords_transform(sino3d, SRTR))
    


phantom = phantom1(1)
nX, nY, nZ = phantom.shape

nAng = 256
Angs = np.linspace(0,np.pi*2,nAng,endpoint=False)


#Sinograms with different wobbles parameters
center=(nX/2.-.5,nY/2.-.5,0.5)

sino0 = sg.forward_project_wobble(phantom, Angs, 0, 0, center=center)
sino0 = vir.rebin(sino0, [nAng, nZ, nX])

angX = 20/180*np.pi
angY = 0
sino20x = sg.forward_project_wobble(phantom, Angs, angX, angY, center=center)
sino20x = vir.rebin(sino20x, [nAng, nZ, nX])

angX = 0
angY = 20/180*np.pi
sino20y = sg.forward_project_wobble(phantom, Angs, angX, angY, center=center)
sino20y = vir.rebin(sino20y, [nAng, nZ, nX])

angX = 20/180*np.pi
angY = 20/180*np.pi
sino20xy = sg.forward_project_wobble(phantom, Angs, angX, angY, center=center)
sino20xy = vir.rebin(sino20xy, [nAng, nZ, nX])


#Correct sinograms with different wobbles parameters
nViews, nRows, nCols = sino0.shape
                    
phi = 20/180*np.pi
theta = np.pi/2
center = np.array([0.0,0.5,nCols/2.-.5])
for view in [0,32,64]:
    ang = Angs[view]
    test =  calib_precesion(sino20x, Angs, ang, phi, theta, center)
    plt.imshow((test - sino0[view,:,:]), origin='lower')
    plt.show()


phi = 20/180*np.pi
theta = 0
center = np.array([0.0,0.5,nCols/2.-.5])
for view in [0,32,64]:
    ang = Angs[view]
    test =  calib_precesion(sino20y, Angs, ang, phi, theta, center)
    plt.imshow((test - sino0[view,:,:]), origin='lower')
    plt.show()

#WRONG
phi =np.sqrt((.349)**2*2)
theta = np.pi/4
center = np.array([0.0,0.5,nCols/2.-.5])
for view in [0,32,64]:
    ang = Angs[view]
    test =  calib_precesion(sino20xy, Angs, ang, phi, theta, center)
    plt.imshow((test - sino0[view,:,:]), origin='lower')
    plt.show()


#Sinograms with different detector tilts/shifts
r = 20/180*np.pi
s = 0
sino_dr =  sg.add_detector_tilt_shift(sino0, r, s)
    
r = 0
s = 15.25
sino_ds =  sg.add_detector_tilt_shift(sino0, r, s)

r = 23/180*np.pi
s = 15.25
sino_drs =  sg.add_detector_tilt_shift(sino0, r, s)


#Correct sinograms with different detector tilts/shifts
r = 20/180*np.pi
s = 0
for view in np.arange(16)*16:
    ang = Angs[view]
    test =  calib_det_orient(sino_dr, Angs, ang, r, s)
    plt.imshow((test - sino0[view,:,:]), origin='lower')
    plt.show()

r = 0
s = 15.25
for view in np.arange(16)*16:
    ang = Angs[view]
    test =  calib_det_orient(sino_ds, Angs, ang, r, s)
    plt.imshow((test - sino0[view,:,:]), origin='lower')
    plt.show()

r = 23/180*np.pi
s = 15.25
for view in np.arange(16)*16:
    ang = Angs[view]
    test =  calib_det_orient(sino_drs, Angs, ang, r, s)
    plt.imshow((test - sino0[view,:,:]), origin='lower')
    plt.show()


#Sinograms with different detector tilts/shifts and wobble
r = 20/180*np.pi
s = 0
sino_dr =  sg.add_detector_tilt_shift(sino20x, r, s)
    
r = 0
s = 15.25
sino_ds =  sg.add_detector_tilt_shift(sino20x, r, s)

r = 23/180*np.pi
s = 15.25
sino_drs =  sg.add_detector_tilt_shift(sino20x, r, s)


#Sinograms with different detector tilts/shifts and wobble
r = 23/180*np.pi
s = 5.2
phi = 20/180*np.pi
theta = np.pi/2
center = np.array([0.0,0.5,nCols/2.-.5])

sino_T =  sg.add_detector_tilt_shift(sino20x, r, s)
for view in np.arange(16)*16:
    ang = Angs[view]
    test =  calib_both(sino_T, Angs, ang, phi, theta, center, r, s)    
    plt.imshow((test - sino0[view,:,:]), origin='lower')
    plt.show()


phi = 20/180*np.pi
theta = 0
center = np.array([0.0,0.5,nCols/2.-.5])

sino_T =  sg.add_detector_tilt_shift(sino20y, r, s)
for view in np.arange(16)*16:
    ang = Angs[view]
    test =  calib_both(sino_T, Angs, ang, phi, theta, center, r, s)    
    plt.imshow((test - sino0[view,:,:]), origin='lower')
    plt.show()

















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


