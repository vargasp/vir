#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:39:20 2023

@author: pvargas21
"""

import numpy as np

import matplotlib.pyplot as plt

import vir
import vir.analytic_geom as ag


def pyramid_sa(a1,a2):
   return 4*np.arcsin(sin(a1/2) * sin(a2/2)) 


def xyx_a(ang1, ang2):
    x1,y1,z1 = vir.sph2cart(1, ang1/2, 0)
    x2,y2,z2 = vir.sph2cart(1, -ang1/2, 0)
    x3,y3,z3 = vir.sph2cart(1, 0, ang1/2)
    x4,y4,z4 = vir.sph2cart(1, 0, -ang1/2)
    
    return [x1,x2,x3,x4],[y1,y2,y3,y4], [z1,z2,z3,z4]
    

x  = np.sqrt(3)*np.array([1, 1,-1,-1])
y  = np.sqrt(3)*np.array([1,-1,-1, 1])
z  = np.sqrt(3)*np.array([1,1,1,1])

sa1 = ag.pyramid_sa(x, y, z)


