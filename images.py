#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:57:25 2025

@author: pvargas21
"""

import numpy as np
import matplotlib.pyplot as plt

import vt
import vir
import vir.sino_calibration as sc


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex
        
    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)
    
    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))
    
x = np.linspace(-np.pi, 3*np.pi,500)
plt.plot(x, np.cos(x))
plt.title(r'Multiples of $\pi$')
ax = plt.gca()
ax.grid(True)
ax.set_aspect(1.0)
ax.axhline(0, color='black', lw=2)
ax.axvline(0, color='black', lw=2)
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
plt.show()
    
    
    



p1 = np.load('/Users/pvargas21/Desktop/Wobble/wobble_phantom1.npy')
p2 = np.load('/Users/pvargas21/Desktop/Wobble/wobble_phantom2.npy')

sP1 = np.load('/Users/pvargas21/Desktop/Wobble/sinoP1.npy')


            

nAngs, nRows, nCols = sP1.shape
            


vt.CreateImage(sinoT2[0,:,:], title='Projection View 0',\
           xtitle='Detector Cols',ytitle='Projection Angls $\\frac{1}{2\pi}$ ',\
           coords=(-128,128,0,nAngs),aspect=1)

vt.CreateImage(sinoT1[:,128,:], title='Sinogram Row: 128',\
           xtitle='Detector Cols',ytitle='Projection Angls $\\frac{1}{2\pi}$ ',\
           coords=(-128,128,0,nAngs),aspect=1)

    
    
    
    

            