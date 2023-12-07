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



def xyx_a(ang1, ang2):
    x1,y1,z1 = vir.sph2cart(1, ang1/2, 0)
    x2,y2,z2 = vir.sph2cart(1, -ang1/2, 0)
    x3,y3,z3 = vir.sph2cart(1, 0, ang1/2)
    x4,y4,z4 = vir.sph2cart(1, 0, -ang1/2)
    
    return [x1,x2,x3,x4],[y1,y2,y3,y4], [z1,z2,z3,z4]


    
def print_rad(angle):
    print(f'{angle/np.pi:.5f}\N{GREEK SMALL LETTER PI}')



pa = [0,0,0]
pa = [1,1,0]
pb1 = [1,1,1]
pb2 = [1,-1,1]
pb3 = [-1,-1,1]
pb4 = [-1,1,1]


ag.oblique_pym([1,1,.0], pb1,pb2,pb3,pb4)

print_rad(sa_cone(np.pi))


x  = np.sqrt(3)*np.array([1, 1,-1,-1])
y  = np.sqrt(3)*np.array([1,-1,-1, 1])
z  = np.sqrt(3)*np.array([1,1,1,1])

ang1, ang2 = ag.da_pyrmaid([0,0,0], [1,1,1], [1,-1,1], [-1,-1,1], [-1,1,1])
print_rad(ag.sa_pyramid(ang1,ang2))


sa1 = ag.pyramid_sa(x, y, z)


