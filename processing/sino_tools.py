import numpy as np
import xcompy as xc

def line_ints_mat2energy(line_ints_mats, attenuations):
    """
    Creates as set of energy dependent line integrals from a sinogram material lengths
    
    Parameters:
    line_ints_mats: Array of line integrals (nAngles,nBins) or (nAngles,nBins,nMats) [cm]
    attenuations:   Array of attentuations (nEnergies) or (nMats,nEnergies) [cm^-1]
    
    Returns:
    Array of line integrals (nAngles,nBins,nEnergies) [unitless]
    """
    
    if line_ints_mats.ndim == 2:
        return np.repeat(line_ints_mats[:,:,np.newaxis], attenuations.size, axis=2)*attenuations
    else:
        return np.sum(np.repeat(line_ints_mats[:,:,:,np.newaxis], attenuations.shape[1], axis=3)*attenuations, axis=2)


def line_ints_noiseless(line_ints_energy, spec):
    """
    Creates as set of noiseless line integrals from a an energy dependent line integrals of attenuation
    
    Parameters:
    line_ints_energy: Array of line integrals (nAngles,nBins,nEnergies) [unitless]
    spec:   spectrum class
    
    Returns:
    Array of line integrals (nAngles,nBins,nEnergies) [unitless]
    """
    return np.sum(line_ints_energy*spec.countsNorm,axis=2)


def line_ints_noise(line_ints_energy, spec, reals):
    """
    Creates as set of noiseless line integrals from a an energy dependent line integrals of attenuation

    Parameters:
    line_ints_energy: Array of line integrals (nAngles,nBins,nEnergies) [unitless]
    spec:   spectrum class

    Returns:
    Array of line integrals (nAngles,nBins,nEnergies) [unitless]
    """
    return  np.sum(line_ints_energy*spec.countsNorm,axis=2)


def line_ints2tran(line_ints, spec):
    """
    Creates as transmission sinogram
    
    Parameters:
    line_ints: Array of line integrals (nAngles,nBins) or (nAngles,nBins,nEnergies) [unitless]
    counts:   Number of counts per detector channel (int) or (nEnergies) [photons]
    
    Returns:
    Transmission sinogram array (nAngles,nBins) or (nAngles,nBins,nEnergies) [photons]
    """
    
    return spec.counts*np.exp(-line_ints)


def tran2line_ints(I, I0):

    trans_percent = I/I0

    idx_I0_0 = np.where(I0 == 0.0)
    if idx_I0_0[0].size > 0:
        trans_percent[idx_I0_0] = 1.0


    idx_I_0 = np.where(I == 0.0)
    if idx_I_0[0].size > 0:
        trans_percent[idx_I_0] = np.unique(I)[1]
    
    return -1.0 * np.log(trans_percent)
    
    np.partition(g.flatten(), 1)[1]
    trans_percent[np.where(trans_percent < 1.0)] = 1.0

    return spec.counts*np.exp(-line_ints_energy)

def atten2HU(rec,EffMuWater):
    return 1000.*(rec - EffMuWater)/EffMuWater

def line_ints_energy2sum(line_ints_energy,spec,det_prob=1.0,alpha=1,reals=0):
    '''
    det_prob: the probability the photon will be absorbed by the detector
    alpha: constant of proportionality in energy integratingin (0 for photon counting)
    '''

    if alpha == 0:
        energy_weights = det_prob
    else:
        energy_weights = alpha*spec.energies*det_prob
    
    I0 = np.trapz(spec.counts*energy_weights,x=spec.energies)
    print(I0)
    
    if reals == 0:
        return -1.0*np.log(np.trapz(spec.counts*np.exp(-line_ints_energy)*energy_weights,x=spec.energies, axis=2)/I0)

    else:
        return -1.0*np.log(np.trapz(np.random.poisson(spec.counts, size=np.insert(line_ints_energy.shape,0,reals)) \
            *np.exp(-line_ints_energy)*energy_weights, x=spec.energies, axis=3)/I0)

    
def detector_prob(energies,material,thickness,rho):
    return (1.0 - np.exp(-1.0*xc.mixatten(material,energies)*thickness*rho))


def CreateViewsArray(nViews=1024,coverage=2.0*np.pi,rotation='CCW',ang_offset=0.0):
    '''
    nViews: 1024 - The number of views
    offset: 0.0 - The angular offest. Default is the source position is at -x [radians]
    rotaion: CCW  - the source moves counter clockwise in the cartesian coordiant system.
    coverage: 2 pi - the angular coverage
    '''
    
    return np.linspace(0.0, coverage, nViews, endpoint=False) #Radians

def CreateColsArray(nCols=512,src_det=100.0, dDet=0.1,equispaced=False):
    '''
    nDets: 512 - The number of detectors in a single row
    dDet: 0.1  - The detetector size [cm]
    equispaced: Flat detector geometry. Default is equalangular
    src_det: 100 - Distance between the source and detector [cm]
    '''
    
    if equispaced:
        dCol = float(dDet)/src_det*src_iso #Effective distance of detector at isocenter (units)
    else:
        dCol = float(dDet)/src_det #Fan angle per detector in (radians)

    return np.linspace(1-nCols,nCols-1,nCols)*dCol/2.0
