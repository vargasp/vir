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
import filtration as ft
import proj as pj
import visualization_tools as vt
import vir 



dMinSphere = 0.4 #[Micronns]
SphereScale = dMinSphere/0.8 #[Unitless]
nViews = 1024
dDet = 0.5 #[Micronns]
nDets = (32/dDet)


derenzo_spheres = pt.DerenzoPhantomSpheres(scale=SphereScale)
pd = pt.Phantom(spheres = derenzo_spheres)
g = vir.Geom(nViews=nViews)
s = vir.Source2d()

#Calculates min number of detectorlets
nDets = pd.minDetectorElems(dDet)
nDets[1] -= 1


d1 = vir.Detector2d(nDets=nDets,dDet=dDet)
d2 = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.25)
d3 = vir.Detector2d(nDets=nDets,dDet=dDet,det_lets=5)
d4 = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.25,det_lets=5)
d5 = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.25,det_lets=10)

s1 = pj.createParSinoClass(g,d1,s,pd) # [distance * linear attenuation]
s2 = pj.createParSinoClass(g,d2,s,pd)
s3 = pj.createParSinoClass(g,d3,s,pd)
s4 = pj.createParSinoClass(g,d4,s,pd)
s5 = pj.createParSinoClass(g,d5,s,pd)

f1 = ft.filter_sino(s1,dCol=dDet)
f2 = ft.filter_sino(s2,dCol=dDet)
f3 = ft.filter_sino(s3,dCol=dDet)
f4 = ft.filter_sino(s4,dCol=dDet)
f5 = ft.filter_sino(s5,dCol=dDet)



r1 = pj.bp(f1[:,7,:], g.Views, d1,nPixels=nDets[0], dPixel=dDet)
r2 = pj.bp(f2[:,7,:], g.Views, d2,nPixels=nDets[0], dPixel=dDet)
r3 = pj.bp(f3[:,7,:], g.Views, d3,nPixels=nDets[0], dPixel=dDet)
r4 = pj.bp(f4[:,7,:], g.Views, d4,nPixels=nDets[0], dPixel=dDet)

r5 = pj.bp(f1[:,7,:], g.Views, d1,nPixels=nDets[0]*2, dPixel=dDet/2.0)
r6 = pj.bp(f2[:,7,:], g.Views, d2,nPixels=nDets[0]*2, dPixel=dDet/2.0)
r7 = pj.bp(f3[:,7,:], g.Views, d3,nPixels=nDets[0]*2, dPixel=dDet/2.0)
r8 = pj.bp(f4[:,7,:], g.Views, d4,nPixels=nDets[0]*2, dPixel=dDet/2.0)
r9 = pj.bp(f4[:,7,:], g.Views, d4,nPixels=nDets[0]*8, dPixel=dDet/8.0)
r10 = pj.bp(f5[:,7,:], g.Views, d4,nPixels=nDets[0]*8, dPixel=dDet/8.0)


vt.CreateImage(r1, title="dPix=.5, dLets=1, QDO=False",coords=[-16,16,-16,16])
vt.CreateImage(r2, title="dPix=.5, dLets=1, QDO=True",coords=[-16,16,-16,16])
vt.CreateImage(r3, title="dPix=.5, dLets=5, QDO=False",coords=[-16,16,-16,16])
vt.CreateImage(r4, title="dPix=.5, dLets=5, QDO=True",coords=[-16,16,-16,16])


vt.CreateImage(r5, title="dPix=.25, dLets=1, QDO=False",coords=[-16,16,-16,16])
vt.CreateImage(r6, title="dPix=.25, dLets=1, QDO=True",coords=[-16,16,-16,16])
vt.CreateImage(r7, title="dPix=.25, dLets=5, QDO=False",coords=[-16,16,-16,16])
vt.CreateImage(r8, title="dPix=.25, dLets=5, QDO=True",coords=[-16,16,-16,16])



import blur
from scipy import ndimage

lb050 = blur.LorentzianPSF(0.5,dPixel=dDet)
lb075 = blur.LorentzianPSF(0.75,dPixel=dDet)
lb100 = blur.LorentzianPSF(1.0,dPixel=dDet)

s4_1 = s4.copy()
s4_2 = s4.copy()
s4_3 = s4.copy()

for i in range(1024):
    s4_1[i,:,:] = ndimage.convolve(s4[i,:,:],lb050)
    s4_2[i,:,:] = ndimage.convolve(s4[i,:,:],lb075)
    s4_3[i,:,:] = ndimage.convolve(s4[i,:,:],lb100)
        
 
f4_1 = ft.filter_sino(s4_1,dCol=dDet)
f4_2 = ft.filter_sino(s4_2,dCol=dDet)
f4_3 = ft.filter_sino(s4_3,dCol=dDet)

r9 = pj.bp(f4_1[:,7,:], g.Views, d4,nPixels=nDets[0]*2, dPixel=dDet/2.0)
r10 = pj.bp(f4_2[:,7,:], g.Views, d4,nPixels=nDets[0]*2, dPixel=dDet/2.0)
r11 = pj.bp(f4_3[:,7,:], g.Views, d4,nPixels=nDets[0]*2, dPixel=dDet/2.0)
   

vt.CreateImage(r9, title="Lorentzian Blur: 0.50",coords=[-16,16,-16,16])
vt.CreateImage(r10, title="Lorentzian Blur: 0.75",coords=[-16,16,-16,16])
vt.CreateImage(r11, title="Lorentzian Blur: 1.00",coords=[-16,16,-16,16])
    

vt.CreateImage(r4 - r2, title="Diff: 5 lets - 1 let, QDO True",coords=[-16,16,-16,16])
vt.CreateImage(r3 - r1, title="Diff: 5 lets - 1 let, QDO False",coords=[-16,16,-16,16])
vt.CreateImage(r4 - r1, title="Diff: QDO True - False, 5 lets",coords=[-16,16,-16,16])
vt.CreateImage(r2 - r1, title="Diff: QDO True - False, 1 let",coords=[-16,16,-16,16])


vt.CreateImages([r1,r2,r3,r4])
vt.CreateImages([r5,r6,r7,r8])
vt.CreateImages([r4 - r2,r3 - r1,r4 - r1,r2 - r1])
vt.CreateImages([r9,r10,r11])


vt.CreateTiffImage(r9,outfile="gamma0.50")
vt.CreateTiffImage(r10,outfile="gamma0.75")
vt.CreateTiffImage(r11,outfile="gamma1.00")

    
r2 = pj.bp(f2[:,64,:], g.Views, d,dPixel=.5)


vt.CreatePlot(np.array([s1[0,64,:],s2[0,64,:]]).T)
vt.CreatePlot(np.array([f1[0,64,:],f2[0,64,:]]).T)
vt.CreatePlot(np.array([r1[256,:],r2[256,:]]).T)





plt.imshow(sino[0,:,:])






plt.plot(sino[0,64,:])
plt.plot(f[0,64,:])




f1 = ft.filter_sino(s2)

r1 = pj.bp(f1.T, g.Views, d1)





importlib.reload(pt)


c = vir.CartesianGrid3d()




d1 = vir.Detector2d()
d2 = vir.Detector2d(offset=0.25)
d3 = vir.Detector2d(det_lets=5)
d4 = vir.Detector2d(offset=0.25,det_lets=5)

s1 = pj.createParSino(g,d1,s,pd)[:,256,:]
s2 = pj.createParSino(g,d2,s,pd)[:,256,:]
s3 = pj.createParSinoClass(g,d3,s,pd)[:,256,:]
s4 = pj.createParSinoClass(g,d4,s,pd)[:,256,:]























p = pt.Phantom(spheres =[0,0,0,200])
c = vir.CartesianGrid3d()
s = vir.Source2d()
g = vir.Geom(nViews=1024)
d1 = vir.Detector2d()
d2 = vir.Detector2d(offset=0.25)
d3 = vir.Detector2d(det_lets=2)
d4 = vir.Detector2d(offset=0.25,det_lets=2)

s1 = pj.createParSino(g,d1,s,p)[:,256,:]
s2 = pj.createParSino(g,d2,s,p)[:,256,:]
s3 = pj.createParSinoClass(g,d3,s,p)[:,256,:]
s4 = pj.createParSinoClass(g,d4,s,p)[:,256,:]


vt.CreatePlot(s1[0,:], title="S1")
vt.CreatePlot(s2[0,:], title="S2")
vt.CreatePlot(s3[0,:], title="S3")
vt.CreatePlot(s4[0,:], title="S4")

vt.CreatePlot(s1[0,:] - s3[0,:], title="S1")

f1 = ft.filter_sino(s1)
f2 = ft.filter_sino(s2)

r1 = pj.bp(f1.T, g.Views, d1)
r2 = pj.bp(f2.T, g.Views, d2)


import phantoms as pt
import derenzo as dp
importlib.reload(dp)
importlib.reload(pt)

p1 = dp.DerenzoPhantom(dimensions=(512,512,20))
p2 = dp.DerenzoPhantom(dimensions=(512,512,20))
E = p2.E
E = np.squeeze(E[np.where(E[:,2] ==0),:])


derenzo_spheres =  pt.DerenzoPhantomSpheres(dimensions=(60,60,20))
derenzo_spheres = derenzo_spheres*512/60.

pd = pt.Phantom(spheres = derenzo_spheres)
c = vir.CartesianGrid3d()
s = vir.Source2d()
g = vir.Geom(nViews=1024)
d1 = vir.Detector2d()
d2 = vir.Detector2d(offset=0.25)
d3 = vir.Detector2d(det_lets=5)
d4 = vir.Detector2d(offset=0.25,det_lets=5)

s1 = pj.createParSino(g,d1,s,pd)[:,256,:]
s2 = pj.createParSino(g,d2,s,pd)[:,256,:]
s3 = pj.createParSinoClass(g,d3,s,pd)[:,256,:]
s4 = pj.createParSinoClass(g,d4,s,pd)[:,256,:]



f1 = ft.filter_sino(s1)
f2 = ft.filter_sino(s2)
f3 = ft.filter_sino(s3)
f4 = ft.filter_sino(s4)

r1 = pj.bp(f1.T, g.Views, d1)
r2 = pj.bp(f2.T, g.Views, d2)
r3 = pj.bp(f3.T, g.Views, d3)
r4 = pj.bp(f4.T, g.Views, d4)




vt.CreateImage(r1, title="Clyinder no QDO")
vt.CreateImage(r2, title="Clyinder with QDO")
vt.CreateImage(r1-r2, title="Clyinder Difference")


vt.CreateImage(r1, title="Derenzo Slice no QDO")
vt.CreateImage(r2, title="Derenzo Slice with QDO")
vt.CreateImage(r3, title="Derenzo Slice no QDO, 5 detectorlets")
vt.CreateImage(r4, title="Derenzo Slice with QDO, 5 detectorlets")


vt.CreateImage(r1-r2, title="Derenzo QDO Difference")
vt.CreateImage(r1-r3, title="Derenzo Detlet Difference")
vt.CreateImage(r2-r3, title="Test")

vt.CreatePlot(r3[107:111,190:330].T)


vt.CreatePlot(r3[108,190:330].T)


ys = np.array([r1[108,240:270],r2[108,240:270],r3[108,240:270],r4[108,240:270]]).T
labels = ["Detlets=1", "Detlets=1, QDO","Detlets=5", "Detlets=5, QDO"]
vt.CreatePlot(ys[3:15,:],labels=labels)



vt.CreatePlot(np.array([r1[256,51:60],r2[256,51:60]]).T)




importlib.reload(vir)
plt.imshow(r1)


importlib.reload(vir)
plt.imshow(vir.mask(r1-r2))







import derenzo as dp
importlib.reload(dp)

t = dp.DerenzoPhantom(dimensions=(512,512,20))







importlib.reload(ft)
x,y = ft.ramp2(512,dBin=1,return_freq=True,zero_pad=False)
plt.plot(x,y,"bo")



f1 = ft.ramp(8)
f2 = ft2.sci_filter(s1.T,output_size=8,circle=False).T[0,:]
print(max(f1-f2))

importlib.reload(ft)
importlib.reload(ft2)





f1 = ft.filter_sino(s1)[0,:]
f2 = ft2.sci_filter(s1.T,output_size=8,circle=False).T[0,:]



f1 = ft.filter_sino(s1)
f2 = ft.sci_filter(s1)


b1 = pj.bp(f1.T, g.Views, d1)
b2 = pj.bp(f2.T, g.Views, d2)



plt.plot(s1[0,249:263])
plt.plot(s2[0,249:263])




plt.plot(b1[255,249:263])
plt.plot(b2[255,249:263])


size = 16

n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=np.int), np.arange(size / 2 - 1, 0, -2, dtype=np.int)))
f1 = np.zeros(size)
f1[0] = 1.0/(4.*dBin)
f1[1::2] = -1 / (np.pi * n * dBin) ** 2


dBin=1
f = 8
z = f*2
V1 = np.fft.fftfreq(f,d=dBin)
V2 = np.fft.fftfreq(z,d=dBin)

print(V1*f)
print(V2*16)


f2 = np.zeros([z])
f2[0] = 1./(4.0*dBin**2)
f2[1::2] = -1.0/(dBin*np.pi*V[1::2])**2
print(f1)    
print(f2)




size = 16
dBin=.25
z = size
V1 = np.fft.fftfreq(z,d=1.0/z)
V2 = np.fft.fftfreq(z,d=dBin)

print(V)
print(V2)


plt.imshow(s1-s2)


importlib.reload(ft)
f1 = ft.filter_sino(s1)[0,:]

f1 = ft.fan_ramp(512)
f2 = ft.fan_ramp2(512)
f3 = ft.rampFT(512)



importlib.reload(ft2)
f2 = ft2.sci_filter(s1.T,output_size=512,circle=False).T[0,:]


plt.plot(f1-f2)


int(2 ** np.ceil(np.log2(2 * img_shape))))
z = int(2**np.ceil(np.log2(2*nBins)))




with np.errstate(invalid='ignore'):
    np.sqrt(-1)


np.sqrt(-1)


import derenzo as pt
import ct_classes as ct
import intersection as inter
import d_proj as dp
import importlib
from skimage.transform import iradon
from scipy.interpolate import interp1d


importlib.reload(ct)
importlib.reload(pt)


#g.updateViews((0,np.pi))
g.updateViews([0])


d1 = ct.Detector2d(QDO=False)
s1 = dp.createParSino(g,d1,s,p)[:,256,:]
d2 = ct.Detector2d(QDO=True)
s2 = dp.createParSino(g,d2,s,p)[:,256,:]

r1 = iradon(s1.T, theta=g.Views/np.pi*180, circle=True)
r2 = iradon(s2.T, theta=g.Views/np.pi*180, circle=True)


s3 = np.zeros((2,512))
s4 = np.zeros((2,512))

s3[0,255] = 1
s3[1,255] = 1

s4[0,255] = 1
s4[0,256] = 1






b3 = bp(s3.T, g.Views, d1)
b4 = bp(s4.T, g.Views, d2, offset =0.250)
py.plot(b4[252:260,256])



py.plot(s1[0,:] - b1[:,256])



py.plot(s1[0,:] - s1[1,:])
py.plot(s2[0,:] - s2[1,:])



py.imshow(b)

b1 = bp(s1.T, g.Views, d1)
py.plot(s1[0,:] - b1[:,256])


b2 = bp(s2.T, g.Views, d1)
py.plot(s2[0,:] - b2[:,256])



b2 = bp(s2.T, g.Views, d2)
py.plot(s1[0,:] - b2[:,256])



b1 = bp(s1.T, g.Views, d2)
b2 = bp(s2.T, g.Views, d2)
py.plot(b1[:,256] - b2[:,256])



nBins =16
zpBins = (2**np.ceil(np.log(2.*nBins-1.)/np.log(2.))).astype(int)

coords = np.linspace(0,zpBins-1,zpBins) - zpBins/2.0
freq_absic = np.abs(np.roll(coords/zpBins,(-zpBins//2)))

print(freq_absic)

