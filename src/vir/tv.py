#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 17:03:11 2022

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize
from skimage.restoration import denoise_tv_chambolle

import vir.analytic_geom as ag
import vir.siddon as sd
import vir.projection as proj
import vir
import vt




def pyT2(AA):
    w1 = 1.0
    w2 = 1.0    

    d11 = AA[0,...]*AA[0,...]
    d12 = AA[0,...]*AA[1,...]
    d21 = AA[1,...]*AA[0,...]
    d22 = AA[1,...]*AA[1,...]


    trace = d11+d22;
    det = d11*d22-d12*d21;

    eig_det=0

    eig_det = (0.25*trace*trace-det).clip(0)
    lam1 = 0.5*trace + np.sqrt(eig_det).clip(0)
    lam2 = 0.5*trace - np.sqrt(eig_det).clip(0)


    s1 = np.sqrt(lam1)
    s2 = np.sqrt(lam2)

    #If matrix is outside K, then project it
    #ind1 = np.where(np.max(s1*w1,s2*w2) > 1.0)
    
    #Find orthonormal eigenvectors of A^TA
    v11 = lam1 - d22
    v12 = lam2 - d22
    v21 = d12
    v22 = d12
    v11 /= np.sqrt(v11*v11+v21*v21)
    v12 /= np.sqrt(v12*v12+v22*v22)
    v21 /= np.sqrt(v11*v11+v21*v21)
    v22 /= np.sqrt(v12*v12+v22*v22)

    ind2 = np.where( (d12*d12 == 0.0) & (d11>=d22) )
    v11[ind2] = 1.0
    v21[ind2] = 0.0
    v12[ind2] = 0.0
    v22[ind2] = 1.0
    
    ind2 = np.where( (d12*d12 == 0.0) & (d11<d22) )
    v11[ind2] = 0.0
    v21[ind2] = 1.0
    v12[ind2] = 1.0
    v22[ind2] = 0.0


    #Project s1,s2 onto l-infinity ball
    s1p = np.min(np.full(s1.shape, w1),s1)
    s2p = np.min(np.full(s2.shape, w1),s2)

    #Compute Sigma^+ * Sigma_p
    s1p /= s1
    s2p = np.where(s2 > 0.0, s2p/s2, 0.0)

    #Compute T = V* \Sigma^+ * \Sigma_p * V^T
    t11 = s1p*v11*v11 + s2p*v12*v12
    t12 = s1p*v11*v21 + s2p*v12*v22
    t21 = s1p*v21*v11 + s2p*v22*v12
    t22 = s1p*v21*v21 + s2p*v22*v22
    
    #Result \Pi(A) = A*T
    ai1 = AA[0,...]
    ai2 = AA[1,...]

    AA[0,...] = ai1*t11 + ai2*t21
    AA[1,...] = ai1*t12 + ai2*t22
    
    return AA



def pyT(AA):
    
    w1 = 1.0
    w2 = 1.0    
    n_rows = AA.shape[1]
    n_cols = AA.shape[2]

    for i in range(n_rows):
        for j in range(n_cols):
            ind1 = np.unravel_index(0*n_rows*n_cols+i*n_cols+j, AA.shape)
            ind2 = np.unravel_index(1*n_rows*n_cols+i*n_cols+j, AA.shape)
            
            
            d11 = AA[ind1]*AA[ind1]
            d12 = AA[ind1]*AA[ind2]
            d21 = AA[ind2]*AA[ind1]
            d22 = AA[ind2]*AA[ind2]
        
            trace = d11+d22;
            det = d11*d22-d12*d21;
        
            eig_det=0
        
            eig_det = max(0,0.25*trace*trace-det)
            lam1 = max(0,0.5*trace + np.sqrt(eig_det))
            lam2 = max(0,0.5*trace - np.sqrt(eig_det))
        
            s1 = np.sqrt(lam1)
            s2 = np.sqrt(lam2)
        
            #If matrix is outside K, then project it
            if max(s1*w1,s2*w2) > 1.0:
                #Find orthonormal eigenvectors of A^TA
                if d12*d12 == 0.0:
                    if d11>=d22:
                        v11 = 1.0
                        v21 = 0.0
                        v12 = 0.0
                        v22 = 1.0
                    else:
                        v11 = 0.0
                        v21 = 1.0
                        v12 = 1.0
                        v22 = 0.0
                else:
                    v11 = lam1-d22
                    v21 = d12
                    ll = np.sqrt(v11*v11+v21*v21)
                    v11 /= ll
                    v21 /= ll
                
                    v12 = lam2-d22
                    v22=d12
                    ll = np.sqrt(v12*v12+v22*v22)
                    v12 /= ll
                    v22 /= ll
        
                #Project s1,s2 onto l-infinity ball
                s1p = min(w1*1.0,s1)
                s2p = min(w2*1.0,s2)
        
                #Compute Sigma^+ * Sigma_p
                s1p /= s1
                if s2 > 0.0:
                    s2p = s2p/s2
                else:
                    s2p = 0.0

        
                #Compute T = V* \Sigma^+ * \Sigma_p * V^T
                t11 = s1p*v11*v11 + s2p*v12*v12
                t12 = s1p*v11*v21 + s2p*v12*v22
                t21 = s1p*v21*v11 + s2p*v22*v12
                t22 = s1p*v21*v21 + s2p*v22*v22
                
                    
                #Result \Pi(A) = A*T
                ai1 = AA[ind1]
                ai2 = AA[ind2]

                AA[ind1] = ai1*t11 + ai2*t21
                AA[ind2] = ai1*t12 + ai2*t22


    return AA




def sdlist2sysmat(sdlist,nPixels):
    A = np.zeros((sdlist.size,np.prod(nPixels)))

    c = sd.list_count_elem(sdlist)
    sdlist = sd.list_ravel(sdlist,nPixels)
                 
    for i, v in np.ndenumerate(sdlist):
        for j in range(c[i]):
            A[np.ravel_multi_index(i,sdlist.shape),v[0][j]] = v[1][j]
            
    return A


def grad_abs(x):
    a = x + np.roll(x,-1,0)
    b = x + np.roll(x,-1,1)

    return np.stack([a,b])


def grad(x):
    a = np.roll(x,-1,0) - x
    b = np.roll(x,-1,1) - x

    return np.stack([a,b])

        
def div_abs(x):
    a = x[0,...] + np.roll(x[0,...],1,0)
    b = x[1,...] + np.roll(x[1,...],1,1)

    return a + b    

def div(x):
    a =  x[0,...] - np.roll(x[0,...],1,0)
    b =  x[1,...] - np.roll(x[1,...],1,1)

    return a + b    



#Sheep Lgan Cicular
nPix = 64
nPixels = (nPix,nPix,1)
dPix = 1.0
nDets = nPix
dDet = 1.0
nTheta = nDets+1
det_lets = 1
src_lets = 1


Dets = vir.censpace(nDets,c=0,d=dDet)
DetsDist = nPix*dPix/2.0

dTheta = np.pi/nTheta
Thetas = vir.boundspace(nTheta,c=np.pi)
srcs, trgs = sd.circular_geom_st(Dets, Thetas)
sdlist = sd.siddons(srcs,trgs,nPixels, dPix).squeeze()

srcs, trgs = square_geom_st(Dets, DetsDist, Thetas, sides=2)
sdlist = sd.siddons(srcs,trgs,nPixels, dPix).squeeze()


#A = sdlist2sysmat(sdlist,nPixels)


image = shepp_logan_phantom()
image = resize(image,(nPix,nPix))
image = image

g  = proj.sd_f_proj(image[:,:,np.newaxis], sdlist)



#Initial Guess Image
u0 = np.zeros([nPix,nPix])


u = u0 * 1.0
q = np.zeros(g.shape)
z = np.zeros((2, nPix, nPix))
ubar = np.zeros(u0.shape)

# Initialize step-size parameters
Sigma_z = 1.0/grad_abs(np.ones(u.shape))
Sigma_q = 1.0/proj.sd_f_proj(np.ones(nPixels), sdlist)
Tau = 1.0/( np.squeeze(proj.sd_b_proj(np.ones(g.shape),sdlist,nPixels)) + div_abs(np.ones(z.shape)))


lam = 1
theta = 1
iters = 100
for k in range(iters):
    print(k)
    # dual-variable updates
    z += Sigma_z*grad(ubar)
    z = pyT2(z)
    
    q += Sigma_q*(proj.sd_f_proj(ubar[:,:,np.newaxis], sdlist) - g)
    q /= (1 + Sigma_q*lam)

    # primal-variable update
    u_previous = u*1.0
    
    u += Tau*(div(z) - np.squeeze(proj.sd_b_proj(q,sdlist,nPixels)))
    ubar = u + theta*(u-u_previous)

