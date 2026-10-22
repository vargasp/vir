import numpy as np
import pylab as py

import spectrums as sp

spec_AcrMV = sp.Spectrum("Accuray_detuned",I0=438850)
spec_AcrAT = sp.Spectrum("Accuray_treatment6MV",I0=269664)
spec_Boone = sp.Spectrum("Boone120kV",I0=164430556)

#Prints the energies and counts for the spectra based in energy bins
print(spec_AcrMV.energies)
print(spec_AcrMV.counts)

#Prints the energies and counts for the sampled spectra at specific energies
print(spec_AcrMV.sampled_energies)
print(spec_AcrMV.sampled_counts)

#Prints the the total number of counts
print(spec_AcrMV.total_counts)
print(spec_AcrMV.counts.sum())
print(np.trapz(spec_AcrMV.sampled_counts,x=spec_AcrMV.sampled_energies))
