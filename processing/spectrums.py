import numpy as np
import xcompy as xc
import os

rootpath = os.path.dirname(__file__)
if os.path.exists(rootpath) == False:
    print('unable to find root directory, using:')
    print(rootpath)
 

def LoadSpectrum(SpectrumName):
    if SpectrumName=="Boone120kV":
        spec_array = np.loadtxt(rootpath+'/Data/Spectrums/Boone120kV.csv',delimiter=',',skiprows=1)[:,0:2]
    elif SpectrumName=="Accuray_detuned":
        spec_array = np.loadtxt(rootpath+'/Data/Spectrums/Accuray_detuned.csv',delimiter=',',skiprows=1)[:,0:2]
        spec_array[:,0] *=1000 #Convert to KeV
    elif SpectrumName=="Accuray_treatment6MV":
        spec_array = np.loadtxt(rootpath+'/Data/Spectrums/Accuray_treatment6MV.csv',delimiter=',',skiprows=1)[:,0:2]
        spec_array[:,0] *=1000 #Convert to KeV
    elif SpectrumName=="Diamond4MV":
        spec_array = np.loadtxt(rootpath+'/Data/Spectrums/Diamond4MV.csv',delimiter=',',skiprows=1)[:,0:2]
        spec_array[:,0] *=1000 #Convert to KeV
    elif SpectrumName[:4]=="Mono":
        spec_array = np.array([[float(SpectrumName[4:]),1e8]])
    else:
        spec_array = np.loadtxt(rootpath+'/Data/Spectrums/' +SpectrumName+ '.csv',delimiter=',')[3:,0:2]

    return np.split(spec_array,2,axis=1)
    

class Spectrum:
    """
    This class creates a spectrum class from an array of sampled counts and energies. The energies can have uniform or non-uniform spacing. Returns energies and counts per bin size (count/Kev * dE)
    """
    def __init__(self,spectrum_name, I0=None):

        self.sampled_energies, self.sampled_counts = LoadSpectrum(spectrum_name)
        self.sampled_energies = np.squeeze(self.sampled_energies) #[KeVs]
        self.sampled_counts = np.squeeze(self.sampled_counts) #[photons @ each energy bin]
        self.uniform_samples = np.unique(self.delta_energies()).size == 1

        if I0 is None:
            self.total_counts = self.sampled_counts_total()
        else:
            self.sampled_counts = self.sampled_counts/self.sampled_counts_total()*I0
            self.total_counts = I0
        
        if self.uniform_samples is True:
            self.energies = self.sampled_energies
            self.norm_counts = self.sampled_counts/self.sampled_counts_total()
        else:
            self.energies = self.mid_energies()
            self.norm_counts = self.mid_counts()/self.sampled_counts_total()
        
        self.counts = self.norm_counts*self.total_counts


    def delta_energies(self):
        if  self.sampled_energies.ndim == 0:
            return 0
        else:
            return self.sampled_energies[1:] - self.sampled_energies[:-1] #[KeVs]

    def mid_energies(self):
        return (self.sampled_energies[1:] + self.sampled_energies[:-1])/2.0 #[KeVs]

    def mid_counts(self):
        return (self.sampled_counts[1:] + self.sampled_counts[:-1])/2.0 * self.delta_energies() #[photons @ each energy bin]

    def effective_energy(self):
        return np.average(self.energies, weights=self.norm_counts) #[KeVs]
    
    def effective_mu_water(self):
        return xc.mixatten('H2O',[self.effective_energy()])[0] #[cm^2/g]

    def sampled_counts_total(self):
        if self.uniform_samples is True:
            return np.sum(self.sampled_counts)
        else:
            return np.trapz(self.sampled_counts,x=self.sampled_energies)

    def Filter(self,material,thickness,normalize=True):
        rho = xc.getRho(material)
        atten = xc.mixatten(material,self.energies)

        counts_filtered = self.counts*np.exp(-1.0*atten*thickness*rho)
        countsTotal_filtered = np.trapz(counts_filtered,x=self.energies) #[photons]

        if normalize:
            self.counts = counts_filtered/countsTotal_filtered*self.total_counts
            self.norm_counts = self.counts/self.total_counts #[unitless]
        else:
            self.counts = counts_filtered
            self.total_counts = countsTotal_filtered
            self.norm_counts = self.counts/self.total_counts #[unitless]

    def inter_energies(self,energies):
        self.sampled_counts = np.interp(energies, self.sampled_energies, self.sampled_counts, left=0, right=0)
        self.sampled_energies = energies #[KeVs]
        
        self.uniform_samples = np.unique(self.delta_energies()).size == 1
        self.sampled_counts = self.sampled_counts/self.sampled_counts_total()*self.total_counts
        
        if self.uniform_samples is True:
            self.energies = self.sampled_energies
            self.norm_counts = self.sampled_counts/self.sampled_counts_total()
        else:
            self.energies = self.mid_energies()
            self.norm_counts = self.mid_counts()/self.sampled_counts_total()
        
        self.counts = self.norm_counts*self.total_counts
        
        
class Spectrum_old:
    def __init__(self,SpectrumName, I0=None):

        self.energies, self.counts = LoadSpectrum(SpectrumName)
        self.energies = np.squeeze(self.energies) #[KeVs]
        self.counts = np.squeeze(self.counts) #[photons @ each energy bin]

        if I0==None:
            self.countsTotal = np.trapz(self.counts,x=self.energies) #[photons]
        else:
            self.counts =  self.counts/np.trapz(self.counts,x=self.energies)*I0 #[photons @ each energy bin]
            self.countsTotal = I0 #[photons]
        
        self.countsNorm = self.counts/self.countsTotal #[unitless]


    def EffEnergy(self):
        return np.average(self.energies, weights=self.countsNorm) #[KeVs]
        
        
    def EffMuWater(self):
        return xc.mixatten('H2O',[self.EffEnergy()])[0] #[cm^2/g]


    def Filter(self,material,thickness,normalize=True):
        rho = xc.getRho(material)
        atten = xc.mixatten(material,self.energies)

        counts_filtered = self.counts*np.exp(-1.0*atten*thickness*rho)
        countsTotal_filtered = np.trapz(counts_filtered,x=self.energies) #[photons]

        if normalize:
            self.counts = counts_filtered/countsTotal_filtered*self.countsTotal
            self.countsNorm = self.counts/self.countsTotal #[unitless]
        else:
            self.counts = counts_filtered
            self.countsTotal = countsTotal_filtered
            self.countsNorm = self.counts/self.countsTotal #[unitless]
            
    def DeltaEnergies(self):
        return self.energies[1:] - self.energies[:-1]

    def MidEnergies(self):
        return (self.energies[1:] + self.energies[:-1])/2.0

    def CountsEnergyBins(self):
        return (self.counts[1:] + self.counts[:-1])/2.0 * (self.energies[1:] - self.energies[:-1])
        
