#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 07:28:39 2026

@author: pvargas21
"""



import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600
import numpy as np

import vir.sys_mat.dd as dd
import vir.sys_mat.rd as rd
import vir.sys_mat.pd as pd



def p_run_single(img3d,sinoP,sinoF,sinoC,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,x0,y0,z0,r,
           fp=True,bp=True):
    nx, ny, nz = img3d.shape
    na, nu, nv = sinoC.shape

    img2d = img3d[:,:,int(nz/2)]

    return rd.aw_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)

def p_run(img3d,sinoP,sinoF,sinoC,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,x0,y0,z0,r,
           fp=True,bp=True):
    nx, ny, nz = img3d.shape
    na, nu, nv = sinoC.shape

    img2d = img3d[:,:,int(nz/2)]


    if fp:
        sino1p = dd.dd_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)
        sino2p = rd.aw_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)
        sino3p = rd.aw_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix,joseph=True)
        sino4p = pd.pd_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)
        
        sino1f = dd.dd_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix)
        sino2f = rd.aw_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix)
        sino3f = rd.aw_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix,joseph=True)
        sino4f = pd.pd_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix)
        
        sino1c = dd.dd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
        sino2c = rd.aw_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
        sino3c = rd.aw_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix,joseph=True)
        sino4c = pd.pd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
    
    if bp:
        
        rec1p = dd.dd_bp_par_2d(sinoP,ang_arr,(nx,ny),du=du,su=su,d_pix=d_pix)
        rec2p = rd.aw_bp_par_2d(sinoP,ang_arr,(nx,ny),du=du,su=su,d_pix=d_pix)
        rec3p = rd.aw_bp_par_2d(sinoP,ang_arr,(nx,ny),du=du,su=su,d_pix=d_pix,joseph=True)
        rec4p = pd.pd_bp_par_2d(sinoP,ang_arr,(nx,ny),du=du,su=su,d_pix=d_pix)
        
        rec1f = dd.dd_bp_fan_2d(sinoF,ang_arr,(nx,ny),DSO,DSD,du=du,su=su,d_pix=d_pix)
        rec2f = rd.aw_bp_fan_2d(sinoF,ang_arr,(nx,ny),DSO,DSD,du=du,su=su,d_pix=d_pix)
        rec3f = rd.aw_bp_fan_2d(sinoF,ang_arr,(nx,ny),DSO,DSD,du=du,su=su,d_pix=d_pix,joseph=True)
        rec4f = pd.pd_bp_fan_2d(sinoF,ang_arr,(nx,ny),DSO,DSD,du=du,su=su,d_pix=d_pix)
        
        rec1c = dd.dd_bp_cone_3d(sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du=du,dv=dv,d_pix=d_pix)
        rec2c = rd.aw_bp_cone_3d(sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du=du,dv=dv,d_pix=d_pix)
        rec3c = rd.aw_bp_cone_3d(sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du=du,dv=dv,d_pix=d_pix,joseph=True)
        rec4c = pd.pd_bp_cone_3d(sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du=du,dv=dv,d_pix=d_pix)
        
        
            


def p_images(img3d,sinoP,sinoF,sinoC,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,x0,y0,z0,r,
             ph=False,fp=True,bp=True):
    nx, ny, nz = img3d.shape
    na, nu, nv = sinoC.shape

    img2d = img3d[:,:,int(nz/2)]


    if ph:
        plt.figure(figsize=(4,4))
        plt.subplot(1,1,1)
        plt.imshow((img3d.transpose([1,0,2]))[:,:,int(nz/2)], cmap='gray', aspect='auto', origin='lower')
        plt.title("Image Phantom")
        plt.xlabel("X Pixels")
        plt.ylabel("Y Pixels")
        plt.show()

    if fp:
        print("Forward Projection Images")
        sino1p = dd.dd_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)
        sino2p = rd.aw_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)
        sino3p = rd.aw_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix,joseph=True)
        sino4p = pd.pd_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)
        
        sino1f = dd.dd_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix)
        sino2f = rd.aw_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix)
        sino3f = rd.aw_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix,joseph=True)
        sino4f = pd.pd_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix)
        
        sino1c = dd.dd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
        sino2c = rd.aw_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
        sino3c = rd.aw_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix,joseph=True)
        sino4c = pd.pd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
    
        sinos = [sino1p,sino2p,sino3p,sino4p,
                 sino1f,sino2f,sino3f,sino4f,
                 sino1c[:,:,int(nv/2)],sino2c[:,:,int(nv/2)],sino3c[:,:,int(nv/2)],sino4c[:,:,int(nv/2)]]
        
            
        titles = ["DD Parallel","SD Parallel","JO Parallel","PD Parallel",
                  "DD Fanbeam","SD Fanbeam","JO Fanbeam","PD Fanbeam",
                  "DD Conebeam","SD Conebeam","JO Conebeam","PD Conebeam"]
        plt.figure(figsize=(16,12))
        for i, (sino,title) in enumerate(zip(sinos,titles)):
            plt.subplot(3,4,i+1)
            plt.imshow(sino, cmap='gray', aspect='auto', origin='lower')
            plt.title(title)
            if i % 4 ==0: 
                plt.ylabel("Angle")
            if i > 7:
                plt.xlabel("Detector Bin")
        plt.show()
        
        
        clip_percent=0
        fractions = [0, 1/8, 1/4, 3/8,1/2]
        #fractions = [0, 1/16, 1/8, 3/16,1/4]
        plt.figure(figsize=(20,8))
        for i, fraction in enumerate(fractions):
            plt.subplot(2,len(fractions),i+1)
            plt.plot(sino1p[int(fraction*na),:].clip(2*r*clip_percent), label='DD')
            plt.plot(sino2p[int(fraction*na),:].clip(2*r*clip_percent), label='AW')
            plt.plot(sino3p[int(fraction*na),:].clip(2*r*clip_percent), label='JO')
            plt.plot(sino4p[int(fraction*na),:].clip(2*r*clip_percent), label='PD')
            plt.legend()
            plt.title("Angle "+ str(int(fraction*360))+": u profile - Parallel")
            if i == 0: plt.ylabel("Intensity")
            
        for i, fraction in enumerate(fractions):
            plt.subplot(2,len(fractions),i+6)
            plt.plot(sino1f[int(fraction*na),:].clip(2*r*clip_percent), label='DD')
            plt.plot(sino2f[int(fraction*na),:].clip(2*r*clip_percent), label='AW')
            plt.plot(sino3f[int(fraction*na),:].clip(2*r*clip_percent), label='JO')
            plt.plot(sino4f[int(fraction*na),:].clip(2*r*clip_percent), label='PD')
            plt.xlabel("Detector Bin")
            plt.legend()
            plt.title("Angle "+ str(int(fraction*360))+": u profile - Fanbeam")
            if i == 0: plt.ylabel("Intensity")
        plt.show()
        
        
        fractions = [0, 1/8, 1/4, 3/8,1/2]
        #fractions = [0, 1/16, 1/8, 3/16,1/4]
        plt.figure(figsize=(20,8))
        for i, fraction in enumerate(fractions):
            plt.subplot(2,len(fractions),i+1)
            plt.plot(sino1c[int(fraction*na),:,int(nv/2+z0)], label='C DD')
            plt.plot(sino2c[int(fraction*na),:,int(nv/2+z0)], label='C AW')
            plt.plot(sino3c[int(fraction*na),:,int(nv/2+z0)], label='C JO')
            plt.plot(sino4c[int(fraction*na),:,int(nv/2+z0)], label='C PD')
            plt.legend()
            plt.title("Angle "+ str(int(fraction*360))+": u profile - Conebeam ")
            if i == 0: plt.ylabel("Intensity")
        
        for i, fraction in enumerate(fractions):
            plt.subplot(2,len(fractions),i+6)
            plt.plot(sino1c[int(fraction*na),int(nu/2+z0),:], label='C DD')
            plt.plot(sino2c[int(fraction*na),int(nu/2+z0),:], label='C AW')
            plt.plot(sino3c[int(fraction*na),int(nu/2+z0),:], label='C JO')
            plt.plot(sino4c[int(fraction*na),int(nu/2+z0),:], label='C PD')
            plt.xlabel("Detector Bin")
            plt.legend()
            plt.title("Angle "+ str(int(fraction*360))+": v profile - Conebeam ")
            if i == 0: plt.ylabel("Intensity")
        plt.show()
        
        
        fractions = [0, 1/8, 1/4, 3/8,1/2]
        #fractions = [0, 1/16, 1/8, 3/16,1/4]
        plt.figure(figsize=(20,4))
        for i, fraction in enumerate(fractions):
            plt.subplot(1,len(fractions),i+1)
            plt.plot(sino1f[int(fraction*na),:], label='F DD')
            plt.plot(sino2f[int(fraction*na),:], label='F AW')
            plt.plot(sino3f[int(fraction*na),:], label='F JO')
            plt.plot(sino1c[int(fraction*na),:,int(nv/2)], label='C DD')
            plt.plot(sino2c[int(fraction*na),:,int(nv/2)], label='C AW')
            plt.plot(sino3c[int(fraction*na),:,int(nv/2)], label='C JO')
            plt.plot(sino4c[int(fraction*na),:,int(nv/2)], label='C PD')
            plt.xlabel("Detector Bin")
            plt.legend()
            plt.title("Angle "+ str(int(fraction*360))+": u profile")
        
            if i == 0: plt.ylabel("Intensity")
        plt.show()
        
        
    if bp:
        #sinoP = sinoP[:1,:]
        #sinoF = sinoF[:1,:]
        #sinoC = sinoC[:1,:]
        
        #ang_arr = np.array([ang_arr[0]])
        
        print("Backprojection Images")
        rec1p = dd.dd_bp_par_2d(sinoP,ang_arr,(nx,ny),du=du,su=su,d_pix=d_pix)
        rec2p = rd.aw_bp_par_2d(sinoP,ang_arr,(nx,ny),du=du,su=su,d_pix=d_pix)
        rec3p = rd.aw_bp_par_2d(sinoP,ang_arr,(nx,ny),du=du,su=su,d_pix=d_pix,joseph=True)
        rec4p = pd.pd_bp_par_2d(sinoP,ang_arr,(nx,ny),du=du,su=su,d_pix=d_pix)
        
        rec1f = dd.dd_bp_fan_2d(sinoF,ang_arr,(nx,ny),DSO,DSD,du=du,su=su,d_pix=d_pix)
        rec2f = rd.aw_bp_fan_2d(sinoF,ang_arr,(nx,ny),DSO,DSD,du=du,su=su,d_pix=d_pix)
        rec3f = rd.aw_bp_fan_2d(sinoF,ang_arr,(nx,ny),DSO,DSD,du=du,su=su,d_pix=d_pix,joseph=True)
        rec4f = pd.pd_bp_fan_2d(sinoF,ang_arr,(nx,ny),DSO,DSD,du=du,su=su,d_pix=d_pix)
        
        rec1c = dd.dd_bp_cone_3d(sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
        rec2c = rd.aw_bp_cone_3d(sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
        rec3c = rd.aw_bp_cone_3d(sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix,joseph=True)
        rec4c = pd.pd_bp_cone_3d(sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
        
        

        
        
        #print(rec1c[int(nx/2),int(nx/2),int(nz/2)])
        #print(rec2c[int(nx/2),int(nx/2),int(nz/2)])
        #print(rec3c[int(nx/2),int(nx/2),int(nz/2)])
        #print(rec4c[int(nx/2),int(nx/2),int(nz/2)])
        
        
        recs = [rec1p,rec2p,rec3p,rec4p,
                rec1f,rec2f,rec3f,rec4f,
                rec1c[:,:,int(nz/2)],rec2c[:,:,int(nz/2)],rec3c[:,:,int(nz/2)],rec4c[:,:,int(nz/2)]]
        titles = ["DD Parallel","SD Parallel","JO Parallel","PD Parallel",
                  "DD Fanbeam","SD Fanbeam","JO Fanbeam","PD Fanbeam",
                  "DD Conebeam","SD Conebeam","JO Conebeam","PD Conebeam"]
        
        plt.figure(figsize=(16,12))
        for i, (rec,title) in enumerate(zip(recs,titles)):
            plt.subplot(3,4,i+1)
            plt.imshow(rec.T, cmap='gray', aspect='auto', origin='lower')
            plt.title(title)
            if i % 4 ==0: 
                plt.ylabel("Pixels")
            if i > 7:
                plt.xlabel("Pixels")
        plt.show()
        
        
        recsp = [rec1p,rec2p,rec3p,rec4p]
        recsf = [rec1f,rec2f,rec3f,rec4f]
        recsc = [rec1c,rec2c,rec3c,rec4c]
        
        labels = ["DD","SD","JO","PD"]
        titles = ["P - X Center", "P - Y Center", "P - XY Center", "P  -YX Center"]
        plt.figure(figsize=(16,12))
        plt.subplot(3,4,1)
        for j, rec in enumerate(recsp):
            plt.plot(rec[:,int(ny/2-1):int(ny/2+1)].mean(axis=1), label=labels[j])
        plt.title(titles[0])
        plt.legend()
        
        plt.subplot(3,4,2)
        for j, rec in enumerate(recsp):
            plt.plot(rec[int(nx/2-1):int(nx/2+1),:].mean(axis=0), label=labels[j])
        plt.title(titles[1])
        plt.legend()
        
        plt.subplot(3,4,3)
        for j, rec in enumerate(recsp):
            plt.plot(rec[np.arange(nx),np.arange(ny)], label=labels[j])
        plt.title(titles[2])
        plt.legend()
        
        plt.subplot(3,4,4)
        for j, rec in enumerate(recsp):
            plt.plot(rec[np.arange(nx), np.arange(ny)[::-1]], label=labels[j])
        plt.title(titles[2])
        plt.legend()
        
        
        titles = ["F - X Center", "F - Y Center", "F - XY Center", "F - YX Center"]
        plt.subplot(3,4,5)
        for j, rec in enumerate(recsf):
            plt.plot(rec[:,int(ny/2-1):int(ny/2+1)].mean(axis=1), label=labels[j])
        plt.title(titles[0])
        plt.legend()
        
        plt.subplot(3,4,6)
        for j, rec in enumerate(recsf):
            plt.plot(rec[int(nx/2-1):int(nx/2+1),:].mean(axis=0), label=labels[j])
        plt.title(titles[1])
        plt.legend()
        
        plt.subplot(3,4,7)
        for j, rec in enumerate(recsf):
            plt.plot(rec[np.arange(nx),np.arange(ny)], label=labels[j])
        plt.title(titles[2])
        plt.legend()
        
        plt.subplot(3,4,8)
        for j, rec in enumerate(recsf):
            plt.plot(rec[np.arange(nx), np.arange(ny)[::-1]], label=labels[j])
        plt.title(titles[2])
        plt.legend()
        
        titles = ["C - X Center", "C - Y Center", "C - XY Center", "C - YX Center"]
        plt.subplot(3,4,9)
        for j, rec in enumerate(recsc):
            plt.plot(rec[:,int(ny/2-1):int(ny/2+1),int(nz/2)].mean(axis=1), label=labels[j])
        plt.title(titles[0])
        plt.legend()
        
        plt.subplot(3,4,10)
        for j, rec in enumerate(recsc):
            plt.plot(rec[int(nx/2-1):int(nx/2+1),:,int(nz/2)].mean(axis=0), label=labels[j])
        plt.title(titles[1])
        plt.legend()
        
        plt.subplot(3,4,11)
        for j, rec in enumerate(recsc):
            plt.plot(rec[np.arange(nx),np.arange(ny),int(nz/2)], label=labels[j])
        plt.title(titles[2])
        plt.legend()
        
        plt.subplot(3,4,12)
        for j, rec in enumerate(recsc):
            plt.plot(rec[np.arange(nx), np.arange(ny)[::-1],int(nz/2)], label=labels[j])
        plt.title(titles[2])
        plt.legend()
        plt.show()
        
        
        
        titles = ["C - X Center", "C - Y Center", "C - Z Center"]
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        for j, rec in enumerate(recsc):
            plt.plot(rec[:,int(ny/2),int(nz/2)], label=labels[j])
        plt.title(titles[0])
        plt.legend()
        
        plt.subplot(1,3,2)
        for j, rec in enumerate(recsc):
            plt.plot(rec[int(nx/2),:,int(nz/2)], label=labels[j])
        plt.title(titles[1])
        plt.legend()
        
        plt.subplot(1,3,3)
        for j, rec in enumerate(recsc):
            plt.plot(rec[int(nx/2),int(ny/2),:], label=labels[j])
        plt.title(titles[2])
        plt.legend()
        plt.show
        
        
        
        
        
