#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 21:18:17 2021

@author: vargasp
"""


import importlib

import numpy as np
import matplotlib.pyplot as plt
import visualization_tools as vt
import vir 

from matplotlib.ticker import FormatStrFormatter, MultipleLocator,FuncFormatter


def createHelicalSampleImage(Views, Z, CN=None, title=None,outfile=None, \
                             y360=None, y180=None,base_tick=np.pi/2):
    """
    

    Parameters
    ----------
    Views : TYPE
        DESCRIPTION.
    Z : TYPE
        DESCRIPTION.
    y360 : (2) array_like, optional
        Y locations for the extent of 2pi reconstruction. The default is None.
    y180 : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    print("h2")
    fig, ax = plt.subplots()
    if CN is None:
        ax.plot(Views, Z, "b.")
    else:
        color_l = ["b","g","r","c","m","y"]

        for i, color in enumerate(np.unique(CN)):
            idx = np.where(CN == color)
            ax.scatter(Views[idx], Z[idx], c=color_l[i], label=color, marker=".")
        
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: \
            '{0}$\pi$'.format(val/np.pi) if val !=0 else '0'))
    ax.xaxis.set_major_locator(MultipleLocator(base=base_tick))

    #Creates shaded regions of 2pi reconstrion
    if y180:
        ax.axhspan(y180[0], y180[1], alpha=0.2, color='yellow')
        ax.axhline(y180[0], color='grey')
        ax.axhline(y180[1], color='grey')

    #Creates shaded regions of 2pi reconstrion
    if y360:
        ax.axhspan(y360[0], y360[1], alpha=0.2, color='green')
        ax.axhline(y360[0], color='grey')
        ax.axhline(y360[1], color='grey')


    ax.set_title(title, fontsize=14,fontname='Times New Roman')
    ax.set_xlabel('Angle', fontsize=12,fontname='Times New Roman')
    ax.set_ylabel('Z position', fontsize=12,fontname='Times New Roman')

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
        print("Here")
        createHelicalSampleImage(Views, Z, title=title,outfile=outfile, \
                            y360=(y360l,y360r), y180=(y180l,y180r))
       
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
                             y180=(y180l,y180r))
        else:
            title ='2pi Wrapped Views: Pitch: '+str(g.pitch/d.nH/d.dH)
            outfile = "P2_p" +str(g.pitch)

            createHelicalSampleImage(VN, ZN, CN=CN, title=title,outfile=outfile, \
                             y360=(y360l,y360r), y180=(y180l,y180r))

    #return VN, ZN, CN


def helical_recon_range(g,d):
    y360l = g.Z[g.nAngles-1] + d.H[0]
    y180l = g.Z[int(g.nAngles/2)-1] + d.H[0]
    y360r = g.Z[-g.nAngles] + d.H[-1]
    y180r = g.Z[-int(g.nAngles/2)] + d.H[-1]
    
    return y360l, y360r, y180l, y180r



dDet = 1.0 #[Micronns]
nDets = (8,8)
coverage = 4*np.pi
nAngles = 16
nViews = nAngles*coverage / (2.0*np.pi) + 1
pitches = np.linspace(1,16,481)
pitches = [0.875]


zp = vir.censpace(193,.0625)

d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0,det_lets=10)


g = vir.Geom(nViews=nViews,pitch=7, coverage=coverage,endpoint=True)
#VN, ZN, CN = scatter_samp(g,d, wrap="None")


"""
for p_idx, pitch in enumerate(pitches):
    g = vir.Geom(nViews=nViews,pitch=pitch, coverage=coverage,endpoint=True)
    scatter_samp(g,d, wrap="None")
"""



scatter_samp(g,d, wrap="None")
    



