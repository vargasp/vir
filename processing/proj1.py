#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:44:45 2020

@author: vargasp
"""

import numpy as np
import vir
import intersection as inter
from scipy import interpolate



def back_proj_ray(phantom, sino_val, ray):
    phantom[ray[0], ray[1], ray[2]] += ray[3]*sino_val

    return phantom














def createSino(g,d,s,p):

    nViews = g.nViews
    Views = g.Views
    
    nRows = d.nH
    nCols = d.nW
    nRow_lets = d.nH_lets
    nCol_lets = d.nW_lets
    H_lets = np.reshape(d.H_lets, [d.nH,d.nH_lets])
    W_lets = np.reshape(d.W_lets, [d.nW,d.nW_lets])

    Spheres = p.S
    
    I0 = 1.0
    nEnergies = 1
        
    sino = np.zeros([nViews,nRows,nCols])
    atten = np.zeros([nRow_lets,nCol_lets])

    #Loops over the angle views (radians)
    for view_idx, view in enumerate(Views):
        
        #Loops over the detector elements in the detector
        for row_idx, Row_lets in enumerate(H_lets):
            for col_idx, Col_lets in enumerate(W_lets):
                
                atten[:,:] = 0.0
                #Loops over the detectorlets in an detector element
                for z_idx, z in enumerate(Row_lets):
                            
                    #Loops over objects in the phantom
                    for sphere_idx, sphere in enumerate(Spheres):
                        r, dist =inter.dist((0,0,1,z),sphere[:4])
        
                        if np.isnan(r) == False:
                            S_2d = [sphere[0], sphere[1], r, sphere[4]]

                            atten[z_idx,:] += inter.AnalyticSinoParSphere(S_2d,view,Col_lets)
                                    
                                
                sino[view_idx,row_idx,col_idx] = np.mean(I0*np.exp(-atten))
                
    return sino


def createSinoLets(g,d,s,p,beamlet_ave=True):

    nViews = g.nViews
    Views = g.Views
    
    nRows = d.nH
    nCols = d.nW
    nRow_lets = d.nH_lets
    nCol_lets = d.nW_lets
    H_lets = d.H_lets
    W_lets = d.W_lets

    Spheres = p.S
    
    I0 = 1.0
    nEnergies = 1
        
    lin_atten = np.zeros([nViews,nRows*nRow_lets,nCols*nCol_lets])
    
    #Loops over the detector elements in the detector
    for row_idx, z in enumerate(H_lets):
    
        #Loops over objects in the phantom
        for sphere_idx, sphere in enumerate(Spheres):
            S_2d = inter.SpherePlaneIntersection((0,0,1,z),sphere[:4])[[0,1,3]]
        
            if S_2d[2] != 0.0:
                lin_atten[:,row_idx,:] += inter.AnalyticSinoParSphere(np.append(S_2d,sphere[4]),Views,W_lets)
                                                                 
    if beamlet_ave:
        return I0*np.exp(-lin_atten).reshape(nViews,nRows,nRow_lets,nCols,nCol_lets).mean(4).mean(2)
    else:
        return I0*np.exp(-lin_atten)



def createSinoLets2(g,d,s,p,beamlet_ave=True):

    nViews = g.nViews
    Views = g.Views
    
    nRows = d.nH
    nCols = d.nW
    nRow_lets = d.nH_lets
    nCol_lets = d.nW_lets
    H_lets = d.H_lets
    W_lets = d.W_lets

    Spheres = p.S
    
    I0 = 1.0
    nEnergies = 1
        
    lin_atten = np.zeros([nViews,nRows*nRow_lets,nCols*nCol_lets])
    
    #Loops over objects in the phantom
    for sphere_idx, sphere in enumerate(Spheres):        
        lin_atten += inter.AnalyticSinoParSphere2(sphere[:4],Views,H_lets,W_lets)*sphere[4]
                                                                 
    if beamlet_ave:
        return I0*np.exp(-lin_atten).reshape(nViews,nRows,nRow_lets,nCols,nCol_lets).mean(4).mean(2)
    else:
        return I0*np.exp(-lin_atten)



def createSinoLets3(g,d,s,p,beamlet_ave=True):

    nViews = g.nViews
    Views = g.Views
    zp = g.Z
    
    nRows = d.nH
    nCols = d.nW
    nRow_lets = d.nH_lets
    nCol_lets = d.nW_lets
    H_lets = d.H_lets
    W_lets = d.W_lets

    Spheres = p.S
    
    I0 = 1.0
    nEnergies = 1
        
    lin_atten = np.zeros([nViews,nRows*nRow_lets,nCols*nCol_lets])
    
    #Loops over the views
    for view_idx, view in enumerate(Views):
    
        #Loops over objects in the phantom
        for sphere_idx, sphere in enumerate(Spheres):        
            lin_atten[view_idx,:,:] += inter.AnalyticSinoParSphere2(sphere[:4],view,H_lets + zp[view_idx],W_lets)*sphere[4]

    if beamlet_ave:
        return I0*np.exp(-lin_atten).reshape(nViews,nRows,nRow_lets,nCols,nCol_lets).mean(4).mean(2)
    else:
        return I0*np.exp(-lin_atten)




def createSinoLets4(g,d,s,p,beamlet_ave=True):

    nViews = g.nViews
    Views = g.Views
    zp = g.Z
    
    nRows = d.nH
    nCols = d.nW
    nRow_lets = d.nH_lets
    nCol_lets = d.nW_lets
    H_lets = d.H_lets
    W_lets = d.W_lets

    Spheres = p.S
    
    I0 = 1.0
    nEnergies = 1
        
    lin_atten = np.zeros([nViews,nRows*nRow_lets,nCols*nCol_lets])
    
    #Loops over the views
    for view_idx, view in enumerate(Views):
        print(view)
        zv = H_lets + zp[view_idx]
        idx = np.where( (Spheres[:,2] + Spheres[:,3] > zv[0]) & \
                       (Spheres[:,2] - Spheres[:,3] < zv[-1]) )
    
        #Loops over objects in the phantom
        for sphere_idx, sphere in enumerate(Spheres[idx]):
            lin_atten[view_idx,:,:] += inter.AnalyticSinoParSphere2(sphere[:4],view,zv,W_lets)*sphere[4]

    if beamlet_ave:
        return I0*np.exp(-lin_atten).reshape(nViews,nRows,nRow_lets,nCols,nCol_lets).mean(4).mean(2)
    else:
        return I0*np.exp(-lin_atten)


def createSinoLets5(g,d,s,p,beamlet_ave=True):

    nViews = g.nViews
    Views = g.Views
    zp = g.Z
    
    nRows = d.nH
    nCols = d.nW
    nRow_lets = d.nH_lets
    nCol_lets = d.nW_lets
    H_lets = d.H_lets
    W_lets = d.W_lets

    Spheres = p.S
    
    I0 = 1.0
    nEnergies = 1

    if d.nW_lets == 1 & d.nH_lets == 1: 
        beamlet_ave = False
        
    lin_atten = np.zeros([nViews,nRows*nRow_lets,nCols*nCol_lets])
    
    #Loops over the views
    for view_idx, view in enumerate(Views):
        #print(view, view_idx,zp[view_idx])
        zv = H_lets + zp[view_idx]
        idx = np.where( (Spheres[:,2] + Spheres[:,3] > zv[0]) & \
                       (Spheres[:,2] - Spheres[:,3] < zv[-1]) )
        

        if idx[0].size!= 0:
            s_min = max(min(Spheres[idx][:,2] - Spheres[idx][:,3]), zv[0])
            s_max = min(max(Spheres[idx][:,2] + Spheres[idx][:,3]), zv[-1]) 
            z_idx = np.searchsorted(zv, (s_min, s_max))

            #Loops over objects in the phantom
            for sphere_idx, sphere in enumerate(Spheres[idx]):
                temp = inter.AnalyticSinoParSphere2(sphere[:4],view,zv[z_idx[0]:(z_idx[-1]+1)],W_lets)*sphere[4]
                lin_atten[view_idx,z_idx[0]:(z_idx[-1]+1),:] += inter.AnalyticSinoParSphere2(sphere[:4],view,zv[z_idx[0]:(z_idx[-1]+1)],W_lets)*sphere[4]

                #print(idx,zv[z_idx[0]],zv[z_idx[-1]],lin_atten[view_idx,:,:].max(), sphere[:4])        

            
    if beamlet_ave:
        return I0*np.exp(-lin_atten).reshape(nViews,nRows,nRow_lets,nCols,nCol_lets).mean(4).mean(2)
    else:
        return I0*np.exp(-lin_atten)


def createParSino(g,d,s,p):

    sino = np.zeros([g.nViews,d.nH,d.nW])
    
    
    for i, z in enumerate(d.Z):
    
        for j in range(p.nS):
            r, dist =inter.dist((0,0,1,z),p.S[j,:4])
        
            if np.isnan(r) == False:
                S_2d = [p.S[j,0], p.S[j,1], r, p.S[j,4]]
                sino[:,i,:] += inter.AnalyticSinoParSphere(S_2d,g.Views,d.X)

    return sino


def createParSinoClass(g,d,s,p, let_sino=False):
    """
    Analytically calculates a sinogram
    
    Parameters
    ----------
    g : class
        Geom class from the VIR module
    d : class
        Geom class from the VIR module
    s : class
        Source class from the VIR module
    p : class
        phantom class from the VIR module
    let_sino : Bool
        Flag to returen sinogram that is not averaged over detectorlets
        
    Returns
    -------
    sino : (g.nViews,d.nW,d.nH) np.array
        
    """    

    #Initializes the sinogram
    if let_sino:
        sino = np.zeros([g.nViews,d.H_lets.size,d.W_lets.size])
        
        #Loops over the rows of the detector
        for i, z in enumerate(d.H_lets):
    
            #Loops over ellipsoids in the phantom
            for j in range(p.nS):
                r, dist =inter.dist((0,0,1,z),p.S[j,:4])
        
                if np.isnan(r) == False:
                    S_2d = [p.S[j,0], p.S[j,1], r, p.S[j,4]]
                    sino[:,i,:] += inter.AnalyticSinoParSphere(S_2d,g.Views,d.W_lets)

        return sino
    else:
        sino = np.zeros([g.nViews,d.nH,d.nW])
  
        #Loops over the rows of the detector
        for i, z in enumerate(d.H_lets):
        
            #Assigns the i index if detectorlets are present
            i = int(i/d.nH_lets)
    
            #Loops over ellipsoids in the phantom
            for j in range(p.nS):
                r, dist =inter.dist((0,0,1,z),p.S[j,:4])
        
                if np.isnan(r) == False:
                    S_2d = [p.S[j,0], p.S[j,1], r, p.S[j,4]]
                    sino[:,i,:] += vir.rebin(inter.AnalyticSinoParSphere(S_2d,g.Views,d.W_lets),[g.nViews,d.nW])

        return sino/d.nH_lets


def createParSinoClassInt(g,d,s,p):
    """
    Analytically calculates a sinogram by integration
    
    Parameters
    ----------
    g : class
        Geom class from the VIR module
    d : class
        Geom class from the VIR module
    s : class
        Source class from the VIR module
    p : class
        phantom class from the VIR module
    let_sino : Bool
        Flag to returen sinogram that is not averaged over detectorlets
        
    Returns
    -------
    sino : (g.nViews,d.nW,d.nH) np.array
        
    """    
    sino = np.zeros([g.nViews,d.nH,d.nW])
  
    X = vir.boundspace(d.nW, d=d.dW, c=d.sW)
    Z = vir.boundspace(d.nH, d=d.dH, c=d.sH)
        
    #Loops over the views of the detector
    for i, View in enumerate(g.Views):
        
        #x0 =xo and  y0 NEEDS CALCULATION
        
        
        #Loops over ellipsoids in the phantom
        for j in range(p.nS):
            
            #x0 =xo and  y0 NEEDS CALCULATION
            x0 = p.S[j,1]
            
            sino[i,:,:] += inter.sphere_int_proj2d(X,Z,x0=x0,y0=p.S[j,2],r=p.S[j,3])

    return sino


def bp(sino, theta, d, nPixels=512, dPixel=1.0):

    nViews, nBins = sino.shape

    Pixels = np.linspace(-nPixels+1,nPixels-1,nPixels)*dPixel/2.0
    xpr, ypr = np.meshgrid(Pixels,Pixels)

    # Reconstruct image by interpolation
    reconstructed = np.zeros((nPixels,nPixels))
    #radius = nPixels // 2
#    xpr, ypr = np.mgrid[:img_shape, :img_shape] - radius
    #x = np.arange(nPixels) - nPixels // 2

    for col, angle in zip(sino, theta):
        t = ypr * np.cos(angle) - xpr * np.sin(angle) + d.sW

        reconstructed += np.interp(t, xp=d.X, fp=col, left=0, right=0)

    """
    if circle:
        out_reconstruction_circle = (xpr ** 2 + ypr ** 2) > radius ** 2
        reconstructed[out_reconstruction_circle] = 0.
    """

    return reconstructed * np.pi / (2 * nViews)


def helix_to_par360li(sino, g,d, Slices):
    '''
    if(nrevs le 1) then message, 'Function helix_to_par360li requires at least 2 revolutions'
    if(nrevs eq 2) then begin
    	if(nslices gt 1) then message, 'Only one slice can be caluclated with 2 revolutions'
        z_int_vec ne 0.0) then message, 'Only slice 0.0 can be interpolated with 2 revolutions'
    
    
    ;Error Control -- Slice positions
if((z_int_vec[0] lt z_rev_vec[1] - pitch_per_angle) || (z_int_vec[nslices-1] gt z_rev_vec[n_elements(z_rev_vec)-1])) then begin
	print, 'Rev Intervals:', z_rev_vec
	print, 'Interp Zs:    ', z_int_vec
	print, string_label(z_int_vec[0]) + ' > ' + string_label(z_rev_vec[1])
	print, string_label(z_int_vec[nslices-1]) + ' < ' + string_label(z_rev_vec[n_elements(z_rev_vec)-1])
	message, 'z_int_vec values outside of interpolation range'
endif

    '''

    #Determine the sinogram's dimensions
    nViews, nRows, nCols  = sino.shape

    nSlices = Slices.size
    
    #Create sinograms
    par_sino = np.empty([g.nAngles,nSlices,nCols]) 
    
    zp = g.Z
    
    
    for angle in range(g.nAngles):
        Rows = d.H + zp[angle]
        f = interpolate.interp1d(Rows, sino[angle,:,:], bounds_error=False,fill_value=0.0,axis =0)
        par_sino[angle,:,:] = f(Slices)


    return par_sino




def helix_to_par180li(sino, g,d, Slices):
    '''
    if(nrevs le 1) then message, 'Function helix_to_par360li requires at least 2 revolutions'
    if(nrevs eq 2) then begin
    	if(nslices gt 1) then message, 'Only one slice can be caluclated with 2 revolutions'
        z_int_vec ne 0.0) then message, 'Only slice 0.0 can be interpolated with 2 revolutions'
    
    
    ;Error Control -- Slice positions
if((z_int_vec[0] lt z_rev_vec[1] - pitch_per_angle) || (z_int_vec[nslices-1] gt z_rev_vec[n_elements(z_rev_vec)-1])) then begin
	print, 'Rev Intervals:', z_rev_vec
	print, 'Interp Zs:    ', z_int_vec
	print, string_label(z_int_vec[0]) + ' > ' + string_label(z_rev_vec[1])
	print, string_label(z_int_vec[nslices-1]) + ' < ' + string_label(z_rev_vec[n_elements(z_rev_vec)-1])
	message, 'z_int_vec values outside of interpolation range'
endif

    '''

    #Determine the sinogram's dimensions
    nViews, nRows, nCols  = sino.shape
    nAngles = int(g.nAngles/2)
    nSlices = Slices.size
    
    #Create sinograms
    par_sino = np.empty([nAngles,nSlices,nCols]) 
    
    zp = g.Z
    
    idx = np.where(np.tile(np.repeat([0,1],nAngles),int(g.nRotations)) == 1)[0]
    sino[idx,:,:] = np.flip(sino[idx,:,:],axis = 2)
    
    
    for angle in range(nAngles):
        Rows = d.H + zp[angle]
        f = interpolate.interp1d(Rows, sino[angle,:,:], bounds_error=False,fill_value=0.0,axis =0)
        par_sino[angle,:,:] = f(Slices)

    return par_sino



def helix_to_par360li2(hsino,g,d,Z_int):
    """
    Interpolates the parallel sinogram from a helical sinogram using 2 pi
    interpolation
    
    Parameters
    ----------
    hsino : (g.nViews, g.nH, g.nW) numpy ndarray
        Helical sinogram
    g : class
        Geom class from the VIR module
    d : class
        Geom class from the VIR module
    Z_int : (nZ) numpy ndarray
        The slices at z to be interpolated
        
    Returns
    -------
    psino : (g.nViews,d.nW,d.nH) np.array
        Parallel sinogram interpolated from the a helical sinogram acquistion
        at the Z_int        
    """    
    
    psino = np.zeros((g.nAngles,Z_int.size,d.nW))

    for angle in range(g.nAngles):
    
        view = angle
        Rows = d.H + g.Z[view]
        view += g.nAngles
        while view  < g.nViews:
            Rows = np.append(Rows, d.H + g.Z[view])
            view += g.nAngles

        Z_acq, counts = np.unique(Rows, return_counts=True)

        pview = np.zeros((Z_acq.size,d.nW))

        view = angle
        Rows = d.H + g.Z[view]
        pview[np.searchsorted(Z_acq, Rows),:] = hsino[view,:,:]
        view += g.nAngles
        while view  < g.nViews:
            Rows = d.H + g.Z[view]
            pview[np.searchsorted(Z_acq, Rows),:] += hsino[view,:,:]
            view += g.nAngles

        pview = pview / counts[np.newaxis].T
        f = interpolate.interp1d(Z_acq, pview, bounds_error=False,fill_value=0.0,axis=0)
        psino[angle,:,:] = f(Z_int)

        
    return psino


def helix_to_par180li2(hsino,g,d,Z_int):


    #Determine the sinogram's dimensions
    nAngles = int(g.nAngles/2)
    
    #Create sinograms
    psino = np.zeros((nAngles,Z_int.size,d.nW))
    hsino180 = np.copy(hsino)
    
    idx = np.where(np.tile(np.repeat([0,1],nAngles),int(g.nRotations)) == 1)[0]
    hsino180[idx,:,:] = np.flip(hsino[idx,:,:],axis = 2)

    for angle in range(nAngles):
    
        view = angle
        Rows = d.H + g.Z[view]
        view += nAngles
        while view  < g.nViews:
            Rows = np.append(Rows, d.H + g.Z[view])
            view += nAngles

        Z_acq, counts = np.unique(Rows, return_counts=True)

        pview = np.zeros((Z_acq.size,d.nW))

        view = angle
        Rows = d.H + g.Z[view]
        pview[np.searchsorted(Z_acq, Rows),:] = hsino180[view,:,:]
        view += nAngles
        while view  < g.nViews:
            Rows = d.H + g.Z[view]
            pview[np.searchsorted(Z_acq, Rows),:] += hsino180[view,:,:]
            view += nAngles

        pview = pview / counts[np.newaxis].T
        f = interpolate.interp1d(Z_acq, pview, bounds_error=False,fill_value=0.0,axis=0)
        psino[angle,:,:] = f(Z_int)
        
    return psino

