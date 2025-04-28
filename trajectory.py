#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 09:40:40 2025

@author: pvargas21
"""

import numpy as np

import vir
from scipy.ndimage import map_coordinates


def sino_axl2hel(sinoA, nRowsH, ViewsH, nAngsH, pitch, Z=None):
    #Data Sizes
    nAngsA, nRowsA, nCols = sinoA.shape
    nViewsH = ViewsH.size
    
    #Generates the projection angles array
    Angs = np.linspace(0,2*np.pi,nAngsA+1,endpoint=True)
    RowsH_g,Cols_g = np.split(np.mgrid[0:nRowsH,0:nCols], 2, axis=0)
    Views_g = np.zeros([1,nRowsH,nCols])

    #If Z is not providedcreates a uniform spaced aquisition
    if Z is None:
        dZ = pitch*nRowsH/nAngsH
        c = nRowsA/2 - nRowsH/2
        Z = vir.censpace(nViewsH,c=c,d=dZ)


    #Loops through views mapping axial to helical projections
    sinoH = np.zeros([nViewsH,nRowsH,nCols], dtype=np.float32)
    for i, (z,view) in enumerate(zip(Z,ViewsH)):
        
        #Ensures the angle is [0, 2*pi)
        view = view%(2*np.pi)
        
        #Calculates the angle index by nearest neghbor
        idxA = (np.abs(Angs - view)).argmin()
        
        #If close to view, use the index angle, if not use linear interpolation
        if np.isclose(Angs[idxA],view):
            idxA = np.round(idxA)
        else:
            idxA = np.interp(view, Angs, np.arange(nAngsA+1))
            
        #Maps the coordinates from the axial sinogram to the helical sinogram
        Views_g[:] = idxA            
        sinoH[i,:,:] = map_coordinates(sinoA,(Views_g,RowsH_g+z,Cols_g), \
                                       order=1, mode="constant", cval=0.0)

        #If view is greater than the largest angle, use grid-wrap interpolation
        #This will also apply to column and row edges
        #Clipping is performed so this will be equivalent to grid-constant in
        #the row and column axes
        if ((view > Angs[-2]) & ~np.isclose(view,Angs[-2])):
            
            print("Extrapolation warning:",i,view)
            sinoH[i,:,:] = map_coordinates(sinoA,(Views_g,\
                                                 (RowsH_g+z).clip(0,nRowsA-1),\
                                                 Cols_g.clip(0,nCols-1)), \
                                           order=1, mode="grid-wrap")

    return sinoH



def sino_hel2axl(sinoH, geomH, AngsA, Z):
    nViews, nRows, nCols = sinoH.shape

    AngsA = np.array(AngsA)
    Z = np.array(Z)


    sinoA1 = np.zeros([AngsA.size,Z.size,nCols], dtype=np.float32)
    sinoA2 = np.zeros([AngsA.size,Z.size,nCols], dtype=np.float32)
    
    for i, ang in enumerate(AngsA):
        idxVp, dVp, idxR_Vp, dR_Vp, idxVn, dVn, idxR_Vn, dR_Vn = geomH.interpZ(Z,ang,nRows)
        #print(idxV[0,:], i, ang)
        #print(idxVp,idxVn)
        #print(dVn, dVp)
        
        #Calculates the weight of row interpolation based
        wRp = dR_Vp[:,1] - dR_Vp[:,0]
        wRn = dR_Vn[:,1] - dR_Vn[:,0]
        with np.errstate(invalid='ignore', divide='ignore'):
            wp0  = np.where(wRp==0.0,0.5,((wRp + dR_Vp[:,0])/wRp))[:,np.newaxis]
            wp1  = np.where(wRp==0.0,0.5,((wRp - dR_Vp[:,1])/wRp))[:,np.newaxis]
            wn0  = np.where(wRn==0.0,0.5,((wRn + dR_Vn[:,0])/wRn))[:,np.newaxis]
            wn1  = np.where(wRn==0.0,0.5,((wRn - dR_Vn[:,1])/wRn))[:,np.newaxis]
        
        sinoA1[i,:,:]= sinoH[idxVp[:,0],idxR_Vp[:,0],:]*wp0 + \
                        sinoH[idxVp[:,1],idxR_Vp[:,1],:]*wp1

        sinoA2[i,:,:]= sinoH[idxVn[:,0],idxR_Vn[:,0],:]*wn0 + \
                        sinoH[idxVn[:,1],idxR_Vn[:,1],:]*wn1

    #print(idxVp.shape)
    #print(dVp.shape)
    #print(idxR_Vp.shape)
    #print(dR_Vp.shape)

    return sinoA1, sinoA2



def sino_hel2axl180(sinoH, geomH, AngsA, Z):
    nViews, nRows, nCols = sinoH.shape

    AngsA = np.array(AngsA)
    Z = np.array(Z)


    sinoA1f = np.zeros([AngsA.size,Z.size,nCols], dtype=np.float32)
    sinoA2f = np.zeros([AngsA.size,Z.size,nCols], dtype=np.float32)
    sinoA1b = np.zeros([AngsA.size,Z.size,nCols], dtype=np.float32)
    sinoA2b = np.zeros([AngsA.size,Z.size,nCols], dtype=np.float32)
    
    for i, ang in enumerate(AngsA):
        idxVp, dVp, idxR_Vp, dR_Vp, idxVn, dVn, idxR_Vn, dR_Vn = geomH.interpZ(Z,ang,nRows)
        #print(idxV[0,:], i, ang)
        #print(idxVp,idxVn)
        #print(dVn, dVp)
        
        #Calculates the weight of row interpolation based
        wRp = dR_Vp[:,1] - dR_Vp[:,0]
        wRn = dR_Vn[:,1] - dR_Vn[:,0]
        with np.errstate(invalid='ignore', divide='ignore'):
            wp0  = np.where(wRp==0.0,0.5,((wRp + dR_Vp[:,0])/wRp))[:,np.newaxis]
            wp1  = np.where(wRp==0.0,0.5,((wRp - dR_Vp[:,1])/wRp))[:,np.newaxis]
            wn0  = np.where(wRn==0.0,0.5,((wRn + dR_Vn[:,0])/wRn))[:,np.newaxis]
            wn1  = np.where(wRn==0.0,0.5,((wRn - dR_Vn[:,1])/wRn))[:,np.newaxis]
        
        sinoA1f[i,:,:]= sinoH[idxVp[:,0],idxR_Vp[:,0],:]*wp0 + \
                        sinoH[idxVp[:,1],idxR_Vp[:,1],:]*wp1

        sinoA2f[i,:,:]= sinoH[idxVn[:,0],idxR_Vn[:,0],:]*wn0 + \
                        sinoH[idxVn[:,1],idxR_Vn[:,1],:]*wn1

        idxVp, dVp, idxR_Vp, dR_Vp, idxVn, dVn, idxR_Vn, dR_Vn = geomH.interpZ(Z,ang+np.pi,nRows)
        #print(idxV[0,:], i, ang)
        #print(idxVp,idxVn)
        #print(dVn, dVp)
        
        #Calculates the weight of row interpolation based
        wRp = dR_Vp[:,1] - dR_Vp[:,0]
        wRn = dR_Vn[:,1] - dR_Vn[:,0]
        with np.errstate(invalid='ignore', divide='ignore'):
            wp0  = np.where(wRp==0.0,0.5,((wRp + dR_Vp[:,0])/wRp))[:,np.newaxis]
            wp1  = np.where(wRp==0.0,0.5,((wRp - dR_Vp[:,1])/wRp))[:,np.newaxis]
            wn0  = np.where(wRn==0.0,0.5,((wRn + dR_Vn[:,0])/wRn))[:,np.newaxis]
            wn1  = np.where(wRn==0.0,0.5,((wRn - dR_Vn[:,1])/wRn))[:,np.newaxis]
        
        sinoA1b[i,:,:]= sinoH[idxVp[:,0],idxR_Vp[:,0],:]*wp0 + \
                        sinoH[idxVp[:,1],idxR_Vp[:,1],:]*wp1

        sinoA2b[i,:,:]= sinoH[idxVn[:,0],idxR_Vn[:,0],:]*wn0 + \
                        sinoH[idxVn[:,1],idxR_Vn[:,1],:]*wn1




    #print(idxVp.shape)
    #print(dVp.shape)
    #print(idxR_Vp.shape)
    #print(dR_Vp.shape)

    return sinoA1f, sinoA2f,sinoA1b[...,::-1], sinoA2b[...,::-1]


def sino_hel2hel(sinoAcq, geomAcq, geomInt, nRowsInt):

    nViewsAcq, nRowsAcq, nCols = sinoAcq.shape



    sinoInt1 = np.zeros([geomInt.nViews,nRowsInt,nCols], dtype=np.float32)
    sinoInt2 = np.zeros([geomInt.nViews,nRowsInt,nCols], dtype=np.float32)
    
    for i, viewInt in enumerate(geomInt.Views):
        Z = vir.censpace(nRowsInt) + geomInt.Z[i]
        idxVp, dVp, idxR_Vp, dR_Vp, idxVn, dVn, idxR_Vn, dR_Vn = geomAcq.interpZ(Z,viewInt,nRowsAcq)
        #print(idxV[0,:], i, ang)
        #print(idxVp,idxVn)
        #print(dVn, dVp)
        
        #Calculates the weight of row interpolation based
        wRp = dR_Vp[:,1] - dR_Vp[:,0]
        wRn = dR_Vn[:,1] - dR_Vn[:,0]
        with np.errstate(invalid='ignore', divide='ignore'):
            wp0  = np.where(wRp==0.0,0.5,((wRp + dR_Vp[:,0])/wRp))[:,np.newaxis]
            wp1  = np.where(wRp==0.0,0.5,((wRp - dR_Vp[:,1])/wRp))[:,np.newaxis]
            wn0  = np.where(wRn==0.0,0.5,((wRn + dR_Vn[:,0])/wRn))[:,np.newaxis]
            wn1  = np.where(wRn==0.0,0.5,((wRn - dR_Vn[:,1])/wRn))[:,np.newaxis]
        
        sinoInt1[i,:,:]= sinoAcq[idxVp[:,0],idxR_Vp[:,0],:]*wp0 + \
                        sinoAcq[idxVp[:,1],idxR_Vp[:,1],:]*wp1

        sinoInt2[i,:,:]= sinoAcq[idxVn[:,0],idxR_Vn[:,0],:]*wn0 + \
                        sinoAcq[idxVn[:,1],idxR_Vn[:,1],:]*wn1

    #print(idxVp.shape)
    #print(dVp.shape)
    #print(idxR_Vp.shape)
    #print(dR_Vp.shape)

    return sinoInt1, sinoInt2


def sino_hel2minhel(sino, geomH, nRows, z):

    z = np.array(z)

    idxVMin = np.searchsorted(geomH.Z + (nRows/2-.5),z.min()-1,side='right')
    idxVMax = np.searchsorted(geomH.Z - (nRows/2-.5),z.max()) + 1

    geomH.updateViews(geomH.Views[idxVMin:idxVMax], Z =geomH.Z[idxVMin:idxVMax])  
    return sino[idxVMin:idxVMax,...]


