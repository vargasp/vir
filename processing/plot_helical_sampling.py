#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 21:18:17 2021

@author: vargasp
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

import vir 


def test(d, g, nAngles):
    """
    Calculates the sample locations of a helical trajectory

    Parameters
    ----------
    d : TYPE
        DESCRIPTION.
    g : TYPE
        DESCRIPTION.
    nAngles : TYPE
        DESCRIPTION.

    Returns
    -------
    VN : TYPE
        DESCRIPTION.
    ZN : TYPE
        DESCRIPTION.
    Z_counts : TYPE
        DESCRIPTION.

    """
    VN, ZN, CN = np.array([]), np.array([]), np.array([])

    for angle in range(nAngles):
        
        #Row Z locations at angle
        Rows = d.H + g.Z[angle]

        #Adds additional Z locations at subsequent revolutions
        view = angle + nAngles
        while view  < g.nViews:
            Rows = np.append(Rows, d.H + g.Z[view])
            view += nAngles

        #Unique Z values and counts at angle 
        Z_unique, counts = np.unique(Rows,return_counts=True)   
        
        #View angles 
        VN = np.append(VN ,np.repeat(g.Views[angle], Z_unique.size))
        
        #Z values
        ZN = np.append(ZN, Z_unique)
        Z_counts = np.append(CN, counts)
    
    return VN, ZN, Z_counts


#Sample Zs & Views
def helical_sample1(d, g):
    Views = np.tile(g.Views, [d.H.size,1])
    Zs = np.add.outer(d.H, g.Z)
    
    return Zs, Views


#Sample Zs & Angles
def helical_sample2(d, g, cov=2.0*np.pi):    
    Angles, counts = np.unique(g.Views % cov,return_counts=True)
    
    Zs = np.empty([Angles.size, d.H.size*counts[0]])
    
    for i,angle in enumerate(Angles):
        Zs[i,:] = np.add.outer(g.Z[np.where(g.Views % cov == angle)], d.H).flatten()
    

    return np.sort(Zs, axis=1), Angles




def createHelicalSampleImage(Views, Z, CN=None, title=None,outfile=None, \
                             y360=None, y180=None,base_tick=np.pi/2,xlim=np.pi*2):

    fig, ax = plt.subplots()
    if CN is None:
        ax.plot(Z, Views, "b.")

    else:
        color_l = ["b","g","r","c","m","y"]

        for i, color in enumerate(np.unique(CN)):
            idx = np.where(CN == color)
            ax.scatter(Z[idx],Views[idx], c=color_l[i], label=color, marker=".")
    
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: \
            '{0}$\pi$'.format(val/np.pi) if val !=0 else '0'))
    ax.yaxis.set_major_locator(MultipleLocator(base=base_tick))
    ax.set_ylim(-base_tick/4.,xlim)
    #ax.set_xlim(-20,20)

    #Creates shaded regions of 2pi reconstrion
    if y180:
        ax.axvspan(y180[0], y180[1], alpha=0.2, color='yellow')
        ax.axvline(y180[0], color='grey')
        ax.axvline(y180[1], color='grey')

    #Creates shaded regions of 2pi reconstrion
    if y360:
        ax.axvspan(y360[0], y360[1], alpha=0.2, color='green')
        ax.axvline(y360[0], color='grey')
        ax.axvline(y360[1], color='grey')


    ax.set_title(title, fontsize=14,fontname='Times New Roman')
    ax.set_ylabel('Angle', fontsize=12,fontname='Times New Roman')
    ax.set_xlabel('Z position', fontsize=12,fontname='Times New Roman')

    plt.savefig(outfile +'.pdf',format='pdf',bbox_inches='tight')
    plt.close(fig)


def scatter_samp(g,d, wrap="None"):

    y360l = g.Z[g.nAngles-1] + d.H[0]
    y180l = g.Z[int(g.nAngles/2)-1] + d.H[0]
    y360r = g.Z[-g.nAngles] + d.H[-1]
    y180r = g.Z[-int(g.nAngles/2)] + d.H[-1]
        
    if wrap == "pi":
        nAngles = int(g.nAngles/2)
    else:
        nAngles = g.nAngles
    
    if wrap == "None":
        Views = np.repeat(g.Views,d.nH)
        Z = np.add.outer(g.Z,d.H).flatten()
    
        title = 'Full View: Pitch: '+str(g.pitch/d.nH/d.dH)
        outfile = "P1_p" +str(g.pitch)
        createHelicalSampleImage(Views, Z, title=title,outfile=outfile, \
                            y360=(y360l,y360r), y180=(y180l,y180r), xlim=g.coverage)
       
    else:
        VN, ZN, CN = np.array([]), np.array([]), np.array([])
    
        for angle in range(nAngles):
            Rows = d.H + g.Z[angle]
    
            view = angle + nAngles
            while view  < g.nViews:
                Rows = np.append(Rows, d.H + g.Z[view])
                view += nAngles
    
            ZU, counts = np.unique(Rows,return_counts=True)    
            VN = np.append(VN ,np.repeat(g.Views[angle], ZU.size))
            ZN = np.append(ZN, ZU)
            CN = np.append(CN, counts)

        if wrap == "pi":
            title ='1pi Wrapped Views: Pitch: '+str(g.pitch/d.nH/d.dH)
            outfile = "P3_p" +str(g.pitch)

            createHelicalSampleImage(VN, ZN, CN=CN, title=title,outfile=outfile, \
                             y180=(y180l,y180r),xlim=np.pi,base_tick=np.pi/4.)
        else:
            title ='2pi Wrapped Views: Pitch: '+str(g.pitch/d.nH/d.dH)
            outfile = "P2_p" +str(g.pitch)

            createHelicalSampleImage(VN, ZN, CN=CN, title=title,outfile=outfile, \
                             y360=(y360l,y360r), y180=(y180l,y180r),xlim=np.pi*2)


def helical_recon_range(g,d):
    y360l = g.Z[g.nAngles-1] + d.H[0]
    y180l = g.Z[int(g.nAngles/2)-1] + d.H[0]
    y360r = g.Z[-g.nAngles] + d.H[-1]
    y180r = g.Z[-int(g.nAngles/2)] + d.H[-1]
    
    return y360l, y360r, y180l, y180r






dDet = 1.0
nDets = (8,8)
coverage = 4*np.pi
nAngles = 10
nViews = nAngles*coverage / (2.0*np.pi) 
pitch = 8

d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0)
g = vir.Geom(nViews=nViews,pitch=pitch, coverage=coverage,endpoint=False)


scatter_samp(g,d, wrap="None")
scatter_samp(g,d, wrap="2pi")
scatter_samp(g,d, wrap="pi")
    



