#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:44:45 2020

@author: vargasp
"""

import sys
import numpy as np
from scipy import interpolate

import vir
import intersection as inter
import transformation as tran


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



def fan_backproject_hsieh(Sino, Views, Cols, src_iso, dPixels=(1.0,1.0), nPixels=(512,512), detOff=0.0, pixOff = (0.0,0.0)):
    '''
    x_center, y_center  NOT CONSISTENT
    '''

    nViews,nRows,nCols = Sino.shape

    #Determine some important parameters from the array bin_coords
    dCol = Cols[1] - Cols[0]
    Col0 = Cols[0]

    #Multiply filtered projections contained in sino by cos^2
    Sino = Sino*(np.cos(Cols)**2)

    #Pre-compute trigonometric functions
    sinViews = np.sin(Views)
    cosViews = np.cos(Views)

    #Calculate the image coordinates
    X, Y, image = calc_image_arrays(nImages=nRows, nPixels=nPixels, dPixels=dPixels, pixOff=pixOff)
        
    #Back-project
    for l in range(nViews):
        sys.stdout.write('\rBackprojecting sinogram: view ' +str(l+1)+' of ' +str(nViews) +'n')
        sys.stdout.flush()
        
        Y_factor = src_iso - X*cosViews[l] + Y*sinViews[l]
        ColsLI = (np.arctan((X*sinViews[l] + Y*cosViews[l] - detOff)/Y_factor) - Col0)/dCol
        ColsLI_idx = np.floor(ColsLI).astype(int)
        ColsLI_wt = ColsLI - ColsLI_idx
        
        for k in range(nRows):
            image[k,:,:] += ((1.0 - ColsLI_wt)*Sino[l,k,(ColsLI_idx).clip(0,nCols-1)] + \
                ColsLI_wt*Sino[l,k,(ColsLI_idx+1).clip(0,nCols-1)])/(Y_factor**2)

    print('')
    return np.squeeze(image)


def fan_backproject(Sino, Angles, Bins, FocalLength, dPixel=1.0, nPixels=512, offset=0.0, equispaced=False):
    #Determine dimensions of input arrays
    nBins,nAngles = Sino.shape

    image = np.zeros([nPixels, nPixels])

    #Determine some important parameters from the array bin_coords
    dBin = Bins[1]-Bins[0]
    Bin0 = Bins[0]

    cosAng = np.cos(Angles)
    sinAng = np.sin(Angles)

    #Calculate two arrays, equal in size to the reconstructed image, with elements corresponding
    #to the distance of the corresponding pixel from the origin and the angle a line to the pixel from the origin
    #make with the positive x axis, respectively. (These are the polar coordinates of each point in image space.)
    PixelsR = tran.MakeRadiusImage(nPixels,dPixel=dPixel)
    PixelsT = tran.MakePhiImage(nPixels,dPixel).T
    cosPixelsT = np.cos(PixelsT)
    sinPixelsT = np.sin(PixelsT)

    for angle in range(nAngles):
        cosfac = PixelsR*(cosAng[angle]*cosPixelsT + sinAng[angle]*sinPixelsT) - offset
        sinfac = PixelsR*(sinAng[angle]*cosPixelsT - cosAng[angle]*sinPixelsT) + FocalLength 
        
        #Calculate an array L_squared, of size num_image_pixels X num_image_pixels that holds the value of L squared
        #for each pixel for the current source angular position.
        #Calculate an array proj_coord, of size num_image_pixels X num_image_pixels, that holds the values of
        #the detector coordinate at which a ray from each pixel intersects the detector for the current source angular position.
        if(equispaced):
            proj_coord = FocalLength * cosfac / sinfac
            L_squared = sinfac**2 / FocalLength**2
        else:
            proj_coord = np.arctan2(cosfac, sinfac)
            L_squared = sinfac*sinfac + cosfac*cosfac

        #Now calculate an array holding the index of the nearest measured ray less than proj_coord
        #for each pixel at the projection angle in question
        bin_left = (np.floor( (proj_coord - Bin0) / dBin )).astype(int)

        #Compute vectors holding the linear interpolation weights to be applied to the nearest ray larger than gamma_prime
        #(weight_right) and the nearest weight less than gamma_prime (weight_left) for each pixel at this projection
        weight_right = ( proj_coord - ( bin_left*dBin + Bin0) ) / dBin
        weight_left = 1.0 - weight_right

        #Now, we form the contibutions from the sinogram to each pixel by applying the left weights to the bins to the left
        #(less than) the gamm_prim of interest and the right_weights to the bins to the right.
        #The reform command is iused because IDL converts the 2D array gamm_left of bin indices to a 1D array, thus the
        #output of sino(gamm_left,angle) is a 1D array that must be returned to 2D format.

        left_contribution = weight_left*Sino[bin_left.clip(0,nBins-1),angle]
        right_contribution = weight_right*Sino[(bin_left+1).clip(0,nBins-1),angle]

        image = image + (1.0/L_squared)*(left_contribution + right_contribution)
    return image


def project(image, n_rows, pixel_size, fan_angle, FocalLength, Views, nCols):
    
    nViews = Views.size
    sino = np.zeros([nCols, nViews])
    gamma_min = -fan_angle/2.0
    gamma_max = fan_angle/2.0
    gamma_step = (gamma_max - gamma_min)/nCols
    
    cosViews = np.cos(np.pi - Views*np.pi/180.0)
    sinViews = np.sin(np.pi - Views*np.pi/180.0)
    
    tan_gammas = np.zeros([nCols])
    for n in range(nCols):    
        tan_gammas[n] = np.tan((gamma_min + (0.5+n)*gamma_step)*np.pi/180.0)*FocalLength
    
    i = np.linspace(0, n_rows-1, n_rows, dtype=int)
    j = np.linspace(0, n_rows-1, n_rows, dtype=int)
    xi = np.linspace(1-n_rows,n_rows-1,n_rows)*pixel_size/2.0
    yi = -xi

    xs = FocalLength*sinViews
    ys = FocalLength*cosViews

    for n in range(nViews):
        print(n)

        xk = (xi-xs[n])/pixel_size
        yk = (yi-ys[n])/pixel_size
        ji = xs[n]/pixel_size + (n_rows-1.0)/2.0
        ki = (n_rows-1.0)/2.0 - ys[n]/pixel_size
        
        for m in range(nCols):
            xd = tan_gammas[m]*cosViews[n] - xs[n]
            yd = -tan_gammas[m]*sinViews[n] - ys[n]
    
            L = np.sqrt(xd*xd + yd*yd)
    
            if (np.abs(yd) >= np.abs(xd)):
                coef = pixel_size*L/np.abs(yd)
                j_int = ji + yk*xd/yd
            
                i_l = i
                i_r = i
                j_l = np.floor(j_int).astype(int)
                j_r = np.ceil(j_int).astype(int)
                w = j_int - j_l
                v = (j_l>=0) & (j_r <=(n_rows-1))
            else:        
                coef = pixel_size*L/np.abs(xd)
                i_int = ki - xk*yd/xd
            
                i_l = np.floor(i_int).astype(int)
                i_r = np.ceil(i_int).astype(int)
                j_l = j
                j_r = j
                w = i_int - i_l
                v = (i_l >=0) & (i_r <=(n_rows-1))
                
            sino[m,n] += coef*np.sum(image[i_l[v],j_l[v]]*(1.0-w[v]) + image[i_r[v],j_r[v]]*w[v])

    return sino
    

def fan_forwardproject(Images, FocalLength, src_det, nViews, nCols, dPixel=1.0, dDet=0.1, coverage=2.0*np.pi):
    #Views: Array of views [degrees]
    #nDets: The number of detedtor elements 
    #focal_length: distance from source to isocenter [cm]
    #fan_angle: arc length [degrees]
    #image: phantom slice pixels in [1/cm]
    #dPixels [cm]
    #dDet [cm]
    
    #Retruns
    #sino: 
    
    if Images.ndim == 2:
        Images = Images[np.newaxis,:,:]
       
    nImages = Images.shape[0]
    nPixels = Images.shape[1]
    
    sino = np.zeros([nImages,nViews,nCols])

    dCol = dDet/src_det #Radians

    Cols = np.linspace(1-nCols,nCols-1,nCols)*dCol/2.0 #Radians
    Views = np.linspace(0, coverage, nViews, endpoint=False) #Radians

    cosViews = np.cos(np.pi - Views)
    sinViews = np.sin(np.pi - Views)
    tanViews = np.tan(Cols)*FocalLength


    #Pixel Indices
    i = np.linspace(0, nPixels, nPixels, endpoint=False, dtype=int)
    j = np.linspace(0, nPixels, nPixels, endpoint=False, dtype=int)

    #Pixel Coordinates
    xi = np.linspace(1-nPixels,nPixels-1,nPixels)*dPixel/2.0
    yi = -xi
    x0 = xi[0]
    y0 = yi[0]

    #Source positions
    xs = FocalLength*sinViews 
    ys = FocalLength*cosViews

    for n in range(nViews):
        sys.stdout.write('\rProjecting sinogram: view ' +str(n+1)+' of ' +str(nViews) +'n')
        sys.stdout.flush()

        #Distances between source and indices
        xk = (xs[n]-xi)
        yk = (ys[n]-yi)
        
        for m in range(nCols):
            xd = tanViews[m]*cosViews[n] - xs[n]
            yd = -tanViews[m]*sinViews[n] - ys[n]

            L = np.sqrt(xd*xd + yd*yd)
    
            if (np.abs(yd) >= np.abs(xd)):
                coef = dPixel*L/np.abs(yd)
                j_int = (xs[n] - x0 - yk*xd/yd)/dPixel
            
                j_l = np.floor(j_int).astype(int)
                j_r = np.ceil(j_int).astype(int)
                w = j_int - j_l
                v = (j_l>=0) & (j_r <=(nPixels-1))
                for img in range(nImages):
                    image = Images[img,:,:]
                    sino[img, n,m] += coef*np.sum(image[i[v],j_l[v]]*(1.0-w[v]) + image[i[v],j_r[v]]*w[v])
            else:        
                coef = dPixel*L/np.abs(xd)
                i_int = (y0 - ys[n] + xk*yd/xd)/dPixel
            
                i_l = np.floor(i_int).astype(int)
                i_r = np.ceil(i_int).astype(int)
                w = i_int - i_l
                v = (i_l >=0) & (i_r <=(nPixels-1))
                for img in range(nImages):
                    image = Images[img,:,:]
                    sino[img, n,m] += coef*np.sum(image[i_l[v],j[v]]*(1.0-w[v]) + image[i_r[v],j[v]]*w[v])

    print('')
    return sino


def par_forwardproject(Images, FocalLength, src_det, nViews, nCols, dPixel=1.0, dDet=0.1, coverage=2.0*np.pi):
    #Views: Array of views [degrees]
    #nDets: The number of detedtor elements 
    #focal_length: distance from source to isocenter [cm]
    #fan_angle: arc length [degrees]
    #image: phantom slice pixels in [1/cm]
    #dPixels [cm]
    #dDet [cm]
    
    #Retruns
    #sino: 
    
    if Images.ndim == 2:
        Images = Images[np.newaxis,:,:]

    nImages = Images.shape[0]
    nPixels = Images.shape[1]
    
    sino = np.zeros([nImages,nViews,nCols])


    dCol = dDet #cm

    Cols = np.linspace(1-nCols,nCols-1,nCols)*dCol/2.0 #Radians
    Views = np.linspace(0, coverage, nViews, endpoint=False) #Radians

    cosViews = np.cos(np.pi - Views)
    sinViews = np.sin(np.pi - Views)
    tanViews = Cols

    #Pixel Indices
    i = np.linspace(0, nPixels, nPixels, endpoint=False, dtype=int)
    j = np.linspace(0, nPixels, nPixels, endpoint=False, dtype=int)

    #Pixel Coordinates
    xi = np.linspace(1-nPixels,nPixels-1,nPixels)*dPixel/2.0
    yi = -xi
    x0 = xi[0]
    y0 = yi[0]

    #Source coordinates
    xs = FocalLength*sinViews 
    ys = FocalLength*cosViews

    for n in range(nViews):

        #Distances between source and pixels
        xk = (xs[n]-xi)
        yk = (ys[n]-yi)
        
        for m in range(nCols):
            #Distance between source and detector
            xd = tanViews[m]*cosViews[n] - xs[n]
            yd = -tanViews[m]*sinViews[n] - ys[n]

            if (np.abs(yd) >= np.abs(xd)):
                print(n,m)
                j_int = (xs[n] - x0 - yk*xd/yd)/dPixel
                print(j_int)
                print((-cosViews[n]/sinViews[n])/dPixel)
                print(j_int - (-cosViews[n]/sinViews[n])/dPixel)

                j_l = np.floor(j_int).astype(int)
                j_r = np.ceil(j_int).astype(int)
                w = j_int - j_l
                v = (j_l>=0) & (j_r <=(nPixels-1))
                for img in range(nImages):
                    image = Images[img,:,:]
                    sino[img,n,m] += dPixel*np.sum(image[i[v],j_l[v]]*(1.0-w[v]) + image[i[v],j_r[v]]*w[v])
            else:        
                i_int = (y0 - ys[n] + xk*yd/xd)/dPixel
                #i_int = (sinViews[n]/cosViews[n])/dPixel
                #print n,m,'y: ', -ys[n], y0, y0 - ys[n] 
                
                                      
                i_l = np.floor(i_int).astype(int)
                i_r = np.ceil(i_int).astype(int)
                w = i_int - i_l
                v = (i_l >=0) & (i_r <=(nPixels-1))
                for img in range(nImages):
                    image = Images[img,:,:]
                    sino[img,n,m] += dPixel*np.sum(image[i_l[v],j[v]]*(1.0-w[v]) + image[i_r[v],j[v]]*w[v])

 
    return sino



def calc_image_arrays(nImages=1, nPixels=(512,512), dPixels=(1.0,1.0), pixOff=(0.0,0.0)):

    nX, nY = nPixels
    dX, dY = dPixels
    
    image = np.zeros([nImages, nX, nY])

    X = np.linspace(1-nX,nX-1,nX)*dX/2.0  # range of x-values in reconstruction
    Y = np.linspace(1-nY,nY-1,nY)*dY/2.0  # range of y-values in reconstruction        
    X, Y = np.meshgrid(X + pixOff[1], Y + pixOff[0])
    
    return X, Y, image




def par_forwardproject(Images, FocalLength, src_det, nViews, nCols, dPixel=1.0, dDet=0.1, coverage=2.0*np.pi):
    #Views: Array of views [degrees]
    #nDets: The number of detedtor elements
    #focal_length: distance from source to isocenter [cm]
    #fan_angle: arc length [degrees]
    #image: phantom slice pixels in [1/cm]
    #dPixels [cm]
    #dDet [cm]
    
    #Retruns
    #sino:
    
    if Images.ndim == 2:
        Images = Images[np.newaxis,:,:]

    nImages = Images.shape[0]
    nPixels = Images.shape[1]
    
    sino = np.zeros([nImages,nViews,nCols])


    dCol = dDet #cm

    Cols = np.linspace(1-nCols,nCols-1,nCols)*dCol/2.0 #Radians
    Views = np.linspace(0, coverage, nViews, endpoint=False) #Radians

    cosViews = np.cos(np.pi - Views)
    sinViews = np.sin(np.pi - Views)
    tanViews = Cols

    #Pixel Indices
    i = np.linspace(0, nPixels, nPixels, endpoint=False, dtype=int)
    j = np.linspace(0, nPixels, nPixels, endpoint=False, dtype=int)

    #Pixel Coordinates
    xi = np.linspace(1-nPixels,nPixels-1,nPixels)*dPixel/2.0
    yi = -xi
    x0 = xi[0]
    y0 = yi[0]

    #Source coordinates
    xs = FocalLength*sinViews
    ys = FocalLength*cosViews

    for n in range(nViews):
        print(n)

        #Distances between source and pixels
        xk = (xs[n]-xi)
        yk = (ys[n]-yi)
        
        for m in range(nCols):
            #Distance between source and detector
            xd = tanViews[m]*cosViews[n] - xs[n]
            yd = -tanViews[m]*sinViews[n] - ys[n]

            if (np.abs(yd) >= np.abs(xd)):
                print(n,m)
                j_int = (xs[n] - x0 - yk*xd/yd)/dPixel
                print(j_int)
                print((-cosViews[n]/sinViews[n])/dPixel)
                print(j_int - (-cosViews[n]/sinViews[n])/dPixel)

                j_l = np.floor(j_int).astype(int)
                j_r = np.ceil(j_int).astype(int)
                w = j_int - j_l
                v = (j_l>=0) & (j_r <=(nPixels-1))
                for img in range(nImages):
                    image = Images[img,:,:]
                    sino[img,n,m] += dPixel*np.sum(image[i[v],j_l[v]]*(1.0-w[v]) + image[i[v],j_r[v]]*w[v])
            else:
                i_int = (y0 - ys[n] + xk*yd/xd)/dPixel
                #i_int = (sinViews[n]/cosViews[n])/dPixel
                #print n,m,'y: ', -ys[n], y0, y0 - ys[n]
                
                                      
                i_l = np.floor(i_int).astype(int)
                i_r = np.ceil(i_int).astype(int)
                w = i_int - i_l
                v = (i_l >=0) & (i_r <=(nPixels-1))
                for img in range(nImages):
                    image = Images[img,:,:]
                    sino[img,n,m] += dPixel*np.sum(image[i_l[v],j[v]]*(1.0-w[v]) + image[i_r[v],j[v]]*w[v])

 
    return sino


def fan_forwardproject_lets(Images, FocalLength, src_det, nViews, nCols, nSrcLets, dPixel=1.0, dDet=0.1, coverage=2.0*np.pi):
    #Views: Array of views [degrees]
    #nDets: The number of detedtor elements
    #focal_length: distance from source to isocenter [cm]
    #fan_angle: arc length [degrees]
    #image: phantom slice pixels in [1/cm]
    #dPixels [cm]
    #dDet [cm]
    
    #Retruns
    #sino:
    
    if Images.ndim == 2:
        Images = Images[np.newaxis,:,:]
       
    nImages = Images.shape[0]
    nPixels = Images.shape[1]
    
    sino = np.zeros([nImages,nViews,nCols])

    dCol = dDet/src_det #Radians

    Cols = np.linspace(1-nCols,nCols-1,nCols)*dCol/2.0 #Radians
    Views = np.linspace(0, coverage, nViews, endpoint=False) #Radians

    cosViews = np.cos(np.pi - Views)
    sinViews = np.sin(np.pi - Views)
    tanViews = np.tan(Cols)*FocalLength


    #Pixel Indices
    i = np.linspace(0, nPixels, nPixels, endpoint=False, dtype=int)
    j = np.linspace(0, nPixels, nPixels, endpoint=False, dtype=int)

    #Pixel Coordinates
    xi = np.linspace(1-nPixels,nPixels-1,nPixels)*dPixel/2.0
    yi = -xi
    x0 = xi[0]
    y0 = yi[0]

    #Loop over sourcelets
    dSrc = .20 #Src width
    
    #Source positions
    SrcLets = np.linspace(1-nSrcLets,nSrcLets-1,nSrcLets)*dSrc/nSrcLets/2.0
    Gammas = np.arctan2(SrcLets,FocalLength)
    ys = FocalLength*np.cos(np.add.outer((np.pi - Views),Gammas))/np.cos(Gammas) #(nViews, nSrcLets)
    xs = FocalLength*np.sin(np.add.outer((np.pi - Views),Gammas))/np.cos(Gammas) #(nViews, nSrcLets)
    
    
    for slet in range(nSrcLets):
        print("Sourcelet:", slet)
        
        for n in range(nViews):
            print(n)

            #Distances between source and indices
            xk = (xs[n,slet]-xi)
            yk = (ys[n,slet]-yi)
        
            for m in range(nCols):
                xd = tanViews[m]*cosViews[n] - xs[n,slet]
                yd = -tanViews[m]*sinViews[n] - ys[n,slet]

                L = np.sqrt(xd*xd + yd*yd)
    
                if (np.abs(yd) >= np.abs(xd)):
                    coef = dPixel*L/np.abs(yd)
                    j_int = (xs[n,slet] - x0 - yk*xd/yd)/dPixel
            
                    j_l = np.floor(j_int).astype(int)
                    j_r = np.ceil(j_int).astype(int)
                    w = j_int - j_l
                    v = (j_l>=0) & (j_r <=(nPixels-1))
                    for img in range(nImages):
                        image = Images[img,:,:]
                        sino[img, n,m] += coef*np.sum(image[i[v],j_l[v]]*(1.0-w[v]) + image[i[v],j_r[v]]*w[v])
                else:
                    coef = dPixel*L/np.abs(xd)
                    i_int = (y0 - ys[n,slet] + xk*yd/xd)/dPixel
            
                    i_l = np.floor(i_int).astype(int)
                    i_r = np.ceil(i_int).astype(int)
                    w = i_int - i_l
                    v = (i_l >=0) & (i_r <=(nPixels-1))
                    for img in range(nImages):
                        image = Images[img,:,:]
                        sino[img, n,m] += coef*np.sum(image[i_l[v],j[v]]*(1.0-w[v]) + image[i_r[v],j[v]]*w[v])

    return sino

