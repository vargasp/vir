import numpy as np
import sino_tools as st
import beamhard as bh
import projection as pj
import xcompy as xc
import pylab as py
import roi_tools as rt
import reconstruction as rc
import visualization_tools as vt
import spectrums as sp



def beamhard_cor(line_ints, coefs, mu):
    '''
    Corrects for beamhardening (water only)
    
    line_ints: logged sinogram data
    coefs: array of beam hardening coefficients
    '''
    line_ints = line_ints/mu
    
    return mu*(coefs[0]*line_ints**2+coefs[1]*line_ints + coefs[2])


def beamhard_cor2(line_ints, coefs, mu):
    return line_ints*np.interp(line_ints/mu,np.linspace(0,40,4001),coefs)


def beamhard_test(path_lengths, energies, counts):
    I0 = np.sum(counts)
    
    attens = xc.mixatten('H2O',energies)
    energy_eff = np.average(energies, weights=counts)
    atten_eff = xc.mixatten('H2O', [energy_eff])[0]
    
    sino_p = np.zeros(path_lengths.shape)
    for i, intensity in enumerate(counts):
        sino_p += intensity*np.exp(-1.0*attens[i]*path_lengths)

    path_lengths_p = -1.0*np.log(sino_p/I0)/atten_eff

    coefs = np.polyfit(path_lengths_p,path_lengths,2)
    ys = np.array([path_lengths,path_lengths_p,coefs[0]*path_lengths_p**2+coefs[1]*path_lengths_p + coefs[2]]).T
    
    return coefs, ys

    
def beamhard_test2(path_lengths, energies, counts):
    I0 = np.sum(counts)
    
    attens = xc.mixatten('H2O',energies)
    energy_eff = np.average(energies, weights=counts)
    atten_eff = xc.mixatten('H2O', [energy_eff])[0]
    
    sino_p = np.zeros(path_lengths.shape)
    for i, intensity in enumerate(counts):
        sino_p += intensity*np.exp(-1.0*attens[i]*path_lengths)

    path_lengths_p = -1.0*np.log(sino_p/I0)/atten_eff

    coefs = path_lengths/path_lengths_p
    ys = np.array([path_lengths,path_lengths_p,path_lengths_p*coefs]).T

    return coefs, ys
    

def water_tub_phantom(npixels=512,dpixel=.1,tub_radius=20):
    #npixels: dimension of phantom [pixels]
    #dpixel: pixel size [arbitrary length]
    #tub_radius: length or tub of water's radius [arbirart length]
    
    x,y = np.indices((npixels,npixels))
    return (x - npixels/2)**2 + (y - npixels/2)**2 < (tub_radius/dpixel)**2


def wedge_phantom(npixels=512,dpixel=.1,wedge_side=40):
    phantom = np.empty([npixels,npixels], dtype=bool)
    
    xy0 = int((npixels - wedge_side/dpixel)/2)
    
    for i in range(int(wedge_side/dpixel)):
        for j in range(i,int(wedge_side/dpixel)):
            phantom[xy0+i,xy0+j] = True

    return phantom
    

def phantom2path_lengths(phantom,FocalLength=60., src_det=107.255, nCols=512, dPixel=.1, dDet=0.17988297294911074):
    path_lengths = pj.fan_forwardproject(phantom, FocalLength, src_det, 1, nCols, dPixel=dPixel, dDet=dDet)
    return np.squeeze(path_lengths)




#Phantom Parameters
dPixel = .1 #cm
dSlice = .1 #cm
nPixels = 512
nDets = 512
nViews = 1024
src_iso = 60
src_det = 107.255
FanAngle = 49.2
dDet = FanAngle/nDets*np.pi/180*src_det


specKeV = sp.Spectrum("Boone120kV",I0=164430556)
specMeV = sp.Spectrum("Accuray_detuned",I0=438850)
spec_80Kev = sp.Spectrum("spec80",I0=7.28e8)
spec_140Kev = sp.Spectrum("spec140",I0=2.58e8)
spec_AcrMV = sp.Spectrum("Accuray_detuned",I0=4.39e5)
spec_AcrAT = sp.Spectrum("Accuray_treatment6MV",I0=2.70e5)


phant_wt = water_tub_phantom()
phant_wd = wedge_phantom()

path_lengths_ln = np.linspace(0,40,4001)
path_lengths_wt = phantom2path_lengths(phant_wt)
path_lengths_wd = phantom2path_lengths(phant_wd)

coefs_wt,ys_wt = beamhard_test(path_lengths_wt,specKeV.energies, specKeV.counts)
coefs_wd,ys_wd = beamhard_test(path_lengths_wd,specKeV.energies, specKeV.counts)
coefs_ln,ys_ln = beamhard_test(path_lengths_ln,specKeV.energies, specKeV.counts)
coefs_tb,ys_tb = beamhard_test2(path_lengths_ln,specKeV.energies, specKeV.counts)

vt.CreatePlot(ys_ln, xs=path_lengths_ln, \
title="Correction Coefficient Analysis - Numerical Pathlengths",xtitle="Pathlengths (cm)",ytitle="Pathlengths (cm)",\
labels=["Monoenergetic","Polyenergetic","Polyenergetic - Corrected"],outfile="bh-plt-cor-ln")

vt.CreatePlot(ys_wt, xs=path_lengths_wt, \
title="Correction Coefficient Analysis - Water Tub Phantom",xtitle="Pathlengths (cm)",ytitle="Pathlengths (cm)",\
labels=["Monoenergetic","Polyenergetic","Polyenergetic - Corrected"],outfile="bh-plt-cor-wt")

vt.CreatePlot(ys_wd, xs=path_lengths_wd, \
title="Correction Coefficient Analysis - Wedge Phantom",xtitle="Pathlengths (cm)",ytitle="Pathlengths (cm)",\
labels=["Monoenergetic","Polyenergetic","Polyenergetic - Corrected"],outfile="bh-plt-cor-wd")

vt.CreatePlot(ys_tb, xs=path_lengths_ln, \
title="Correction Coefficient Analysis - Lookup Table",xtitle="Pathlengths (cm)",ytitle="Pathlengths (cm)",\
labels=["Monoenergetic","Polyenergetic","Polyenergetic - Corrected"],outfile="bh-plt-cor-tb")


#Calculate Attenuations
attensWKeV = xc.mixatten('H2O',specKeV.energies)

mat_phant_W100 = rt.mask_circle(nPixels=(nPixels,nPixels),radius=100).astype(np.float)

line_ints_matsW100 = pj.fan_forwardproject(mat_phant_W100, src_iso, src_det, nViews, nDets, dPixel=.1, dDet=dDet)

line_ints_energyW100KeV = st.line_ints_mat2energy(line_ints_matsW100.squeeze(), attensWKeV)

line_ints_W100KeV = st.line_ints_energy2sum(line_ints_energyW100KeV, specKeV,alpha=0)


line_ints_W100KeV_bh_ln = beamhard_cor(line_ints_W100KeV,coefs_ln,specKeV.effmuwater).clip(0)
line_ints_W100KeV_bh_wt = beamhard_cor(line_ints_W100KeV,coefs_wt,specKeV.effmuwater).clip(0)
line_ints_W100KeV_bh_wd = beamhard_cor(line_ints_W100KeV,coefs_wd,specKeV.effmuwater).clip(0)
line_ints_W100KeV_bh_tb = beamhard_cor2(line_ints_W100KeV,coefs_tb,specKeV.effmuwater).clip(0)




rec_W100KeV =  rc.fan_fbp(line_ints_W100KeV, src_iso, src_det, dDet=dDet, nPixels=(512,512), dPixels=(0.1,0.1)).T
rec_W100KeV_ln =  rc.fan_fbp(line_ints_W100KeV_bh_ln, src_iso, src_det, dDet=dDet, nPixels=(512,512), dPixels=(0.1,0.1)).T
rec_W100KeV_wt =  rc.fan_fbp(line_ints_W100KeV_bh_wt, src_iso, src_det, dDet=dDet, nPixels=(512,512), dPixels=(0.1,0.1)).T
rec_W100KeV_wd =  rc.fan_fbp(line_ints_W100KeV_bh_wd, src_iso, src_det, dDet=dDet, nPixels=(512,512), dPixels=(0.1,0.1)).T
rec_W100KeV_tb =  rc.fan_fbp(line_ints_W100KeV_bh_tb, src_iso, src_det, dDet=dDet, nPixels=(512,512), dPixels=(0.1,0.1)).T

rec_W100KeVH = st.atten2HU(rec_W100KeV,specKeV)
rec_W100KeVH_ln = st.atten2HU(rec_W100KeV_ln,specKeV)
rec_W100KeVH_wt = st.atten2HU(rec_W100KeV_wt,specKeV)
rec_W100KeVH_wd = st.atten2HU(rec_W100KeV_wd,specKeV)
rec_W100KeVH_tb = st.atten2HU(rec_W100KeV_tb,specKeV)


vt.CreateImage(rec_W100KeVH, title="Water Tub - No Correction", ctitle="Houndsfield Units",window=(-100,100),outfile="rec_wt")
vt.CreatePlot(rec_W100KeVH[:,256],xs=np.linspace(0,51.2,512),ytitle="Houndsfield Units",xtitle="Distance (cm)",title="Water Tub - No Correction",outfile="plot_wt")

vt.CreateImage(rec_W100KeVH_wt, title="Water Tub - BHC Water Tub", ctitle="Houndsfield Units",window=(-100,100),outfile="rec_wt_wt")
vt.CreatePlot(rec_W100KeVH_wt[:,256],xs=np.linspace(0,51.2,512),ytitle="Houndsfield Units",xtitle="Distance (cm)",title="Water Tub - BHC Water Tub",outfile="plot_wt_wt")

vt.CreateImage(rec_W100KeVH_wd, title="Water Tub - BHC Water Wedge", ctitle="Houndsfield Units",window=(-100,100),outfile="rec_wd")
vt.CreatePlot(rec_W100KeVH_wd[:,256],xs=np.linspace(0,51.2,512),ytitle="Houndsfield Units",xtitle="Distance (cm)",title="Water Tub - BHC Water Wedge",outfile="plot_wt_wd")

vt.CreateImage(rec_W100KeVH_ln, title="Water Tub - BHC Numerical Pathlengths", ctitle="Houndsfield Units",window=(-100,100),outfile="rec_wt_ln")
vt.CreatePlot(rec_W100KeVH_ln[:,256],xs=np.linspace(0,51.2,512),ytitle="Houndsfield Units",xtitle="Distance (cm)",title="Water Tub - BHC Numerical Pathlengths",outfile="plot_wt_ln")

vt.CreateImage(rec_W100KeVH_tb, title="Water Tub - BHC Table Lookup", ctitle="Houndsfield Units",window=(-100,100),outfile="rec_wt_tb")
vt.CreatePlot(rec_W100KeVH_tb[:,256],xs=np.linspace(0,51.2,512),ytitle="Houndsfield Units",xtitle="Distance (cm)",title="Water Tub - BHC Table Lookup",outfile="plot_wt_tb")

attens = xcat_prostate_attens(materials,specKeV.energies)
materials = xcat_prostate_materials()
mt = np.load(infile_folder + 'material_phat.npy')
mat_phant_XCAT = xcat_prostate_material_sino(mt)

line_ints_mats = pj.fan_forwardproject(mat_phant_XCAT, src_iso, src_det, nViews, nDets, dPixel=.1, dDet=dDet).transpose([1,2,0])
line_ints_energy = st.line_ints_mat2energy(line_ints_mats, attens)
line_ints = st.line_ints_energy2sum(line_ints_energy, specKeV,alpha=0)

line_ints_XCAT_bh_ln = beamhard_cor(line_ints,coefs_ln,specKeV.effmuwater).clip(0)
line_ints_XCAT_bh_wt = beamhard_cor(line_ints,coefs_wt,specKeV.effmuwater).clip(0)
line_ints_XCAT_bh_wd = beamhard_cor(line_ints,coefs_wd,specKeV.effmuwater).clip(0)
line_ints_XCAT_bh_tb = beamhard_cor2(line_ints,coefs_tb,specKeV.effmuwater).clip(0)

rec_XCAT =  rc.fan_fbp(line_ints, src_iso, src_det, dDet=dDet, nPixels=(512,512), dPixels=(0.1,0.1)).T
rec_XCAT_ln =  rc.fan_fbp(line_ints_XCAT_bh_ln, src_iso, src_det, dDet=dDet, nPixels=(512,512), dPixels=(0.1,0.1)).T
rec_XCAT_wt =  rc.fan_fbp(line_ints_XCAT_bh_wt, src_iso, src_det, dDet=dDet, nPixels=(512,512), dPixels=(0.1,0.1)).T
rec_XCAT_wd =  rc.fan_fbp(line_ints_XCAT_bh_wd, src_iso, src_det, dDet=dDet, nPixels=(512,512), dPixels=(0.1,0.1)).T
rec_XCAT_tb =  rc.fan_fbp(line_ints_XCAT_bh_tb, src_iso, src_det, dDet=dDet, nPixels=(512,512), dPixels=(0.1,0.1)).T

rec_XCATH = st.atten2HU(rec_XCAT,specKeV)
rec_XCATH_ln = st.atten2HU(rec_XCAT_ln,specKeV)
rec_XCATH_wt = st.atten2HU(rec_XCAT_wt,specKeV)
rec_XCATH_wd = st.atten2HU(rec_XCAT_wd,specKeV)
rec_XCATH_tb = st.atten2HU(rec_XCAT_tb,specKeV)


vt.CreateImage(rec_XCATH, title="Water Tub - No Correction", ctitle="Houndsfield Units",window=(-100,100),outfile="rec_wt")
vt.CreatePlot(rec_XCATH[:,256],xs=np.linspace(0,51.2,512),ytitle="Houndsfield Units",xtitle="Distance (cm)",title="Water Tub - No Correction",outfile="plot_wt")

vt.CreateImage(rec_XCATH_wt, title="Water Tub - BHC Water Tub", ctitle="Houndsfield Units",window=(-100,100),outfile="rec_wt_wt")
vt.CreatePlot(rec_XCATH_wt[:,256],xs=np.linspace(0,51.2,512),ytitle="Houndsfield Units",xtitle="Distance (cm)",title="Water Tub - BHC Water Tub",outfile="plot_wt_wt")

vt.CreateImage(rec_XCATH_wd, title="Water Tub - BHC Water Wedge", ctitle="Houndsfield Units",window=(-100,100),outfile="rec_wd")
vt.CreatePlot(rec_XCATH_wd[:,256],xs=np.linspace(0,51.2,512),ytitle="Houndsfield Units",xtitle="Distance (cm)",title="Water Tub - BHC Water Wedge",outfile="plot_wt_wd")

vt.CreateImage(rec_XCATH_ln, title="Water Tub - BHC Numerical Pathlengths", ctitle="Houndsfield Units",window=(-100,100),outfile="rec_wt_ln")
vt.CreatePlot(rec_XCATH_ln[:,256],xs=np.linspace(0,51.2,512),ytitle="Houndsfield Units",xtitle="Distance (cm)",title="Water Tub - BHC Numerical Pathlengths",outfile="plot_wt_ln")

vt.CreateImage(rec_XCATH_tb, title="Water Tub - BHC Table Lookup", ctitle="Houndsfield Units",window=(-100,100),outfile="rec_wt_tb")
vt.CreatePlot(rec_XCATH_tb[:,256],xs=np.linspace(0,51.2,512),ytitle="Houndsfield Units",xtitle="Distance (cm)",title="Water Tub - BHC Table Lookup",outfile="plot_wt_tb")




mu1 = 0.22288971
mu2 = 0.28510381902750576


line_ints_W100KeV_cor_c1 = beamhard_cor(line_ints_W100KeV,coef_c,1.0)
line_ints_W100KeV_cor_c2 = beamhard_cor(line_ints_W100KeV,coef_c,mu1)
line_ints_W100KeV_cor_c3 = beamhard_cor(line_ints_W100KeV,coef_c,mu2)
line_ints_W100KeV_cor_a1 = beamhard_cor(line_ints_W100KeV,coef_a,1.0)
line_ints_W100KeV_cor_a2 = beamhard_cor(line_ints_W100KeV,coef_a,mu1)
line_ints_W100KeV_cor_a3 = beamhard_cor(line_ints_W100KeV,coef_a,mu2)



r0 = rc.fan_fbp(line_ints_W100KeV, src_iso, src_det, dDet=dDet, nPixels=(512,512),dPixels=(0.1,0.1)).T
r1 = rc.fan_fbp(line_ints_W100KeV_cor_c1, src_iso, src_det, dDet=dDet, nPixels=(512,512),dPixels=(0.1,0.1)).T
r2 = rc.fan_fbp(line_ints_W100KeV_cor_c2, src_iso, src_det, dDet=dDet, nPixels=(512,512),dPixels=(0.1,0.1)).T
r3 = rc.fan_fbp(line_ints_W100KeV_cor_c3, src_iso, src_det, dDet=dDet, nPixels=(512,512),dPixels=(0.1,0.1)).T
r4 = rc.fan_fbp(line_ints_W100KeV_cor_a1, src_iso, src_det, dDet=dDet, nPixels=(512,512),dPixels=(0.1,0.1)).T
r5 = rc.fan_fbp(line_ints_W100KeV_cor_a2, src_iso, src_det, dDet=dDet, nPixels=(512,512),dPixels=(0.1,0.1)).T
r6 = rc.fan_fbp(line_ints_W100KeV_cor_a3, src_iso, src_det, dDet=dDet, nPixels=(512,512),dPixels=(0.1,0.1)).T

#py.plot(r0[:,256],label="0")
#py.plot(r1[:,256],label="1")
py.plot((r2[:,256]/mu1).clip(.8),label="2")
py.plot((r3[:,256]/mu1).clip(.8),label="3")
#py.plot(r4[:,256],label="4")
py.plot((r5[:,256]/mu2).clip(.8),label="5")
py.plot((r6[:,256]/mu2).clip(.8),label="6")
py.legend()
py.show()


print((np.max(r0) - r0[256,256])/r0[256,256], r0[256,256])
print((np.max(r1) - r1[256,256])/r1[256,256], r1[256,256])
print((np.max(r2) - r2[256,256])/r2[256,256], r2[256,256])
print((np.max(r3) - r3[256,256])/r3[256,256], r3[256,256])
print((np.max(r4) - r4[256,256])/r4[256,256], r4[256,256])
print((np.max(r5) - r5[256,256])/r5[256,256], r5[256,256])
print((np.max(r6) - r6[256,256])/r6[256,256], r6[256,256])




print(np.max(line_ints_W100KeV[0,:])/a)
print(np.max(line_ints_W100KeV_cor_a[0,:])/a)
print(np.max(line_ints_W100KeV_cor_c[0,:])/a)

print(np.max(line_ints_W100KeV[0,:])/b)
print(np.max(line_ints_W100KeV_cor_a[0,:])/b)
print(np.max(line_ints_W100KeV_cor_c[0,:])/b)





py.plot(line_ints_W100KeV[0,:])
py.plot(line_ints_W100KeV_cor_a[0,:])
py.plot(line_ints_W100KeV_cor_c[0,:])
py.show()


rec_W100KeVH = st.atten2HU(rec_W100KeV,specKeV)
rec_W100MeVH = st.atten2HU(rec_W100MeV,specMeV)



test = beamhard_test(specKeV.energies, specKeV.counts)





