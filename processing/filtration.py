import numpy as np


def ramp(nBins, dBin=1.0, cutoff=0.5, zero_pad=True):
    """
    Computers a ramp filter based on AC Kak, M Slaney, "Principles of
    Computerized Tomographic Imaging", IEEE Press 1988.
           
    Parameters
    ----------
    nBins : int
        The number of detector columns
    dCol : float, optional
        The sampling interval between detecros [distance].  Default is 1.0
    cutoff : float
        Spatial frequency cut off [cycles/dBin]
    zero_pad : boolean
        If true zero-pads the filter to the smallest power of 2 that is greater
        or equal to (2*nbins) to eliminate the interperiod interference
        artifacts inherent to periodic convolution. Default is True

    Returns
    -------
    ramp_filter: (fBins) ndarray
        The computed ramp filter.

    """
    
    if zero_pad:
        fBins = int(2**np.ceil(np.log2(2*nBins)))
    else:
        fBins = nBins

    V = np.fft.fftfreq(fBins,d=1.0/fBins)
                          
    ramp_filter = np.zeros(fBins)
    ramp_filter[0] = 1./(4.0*dBin**2)
    ramp_filter[1::2] = -1.0/(dBin*np.pi*V[1::2])**2
    
    return 2 * np.real(np.fft.fft(ramp_filter))


def ramp2(nBins, dBin=1.0, cutoff=0.5, zero_pad=True, return_freq=False):
    
    if zero_pad:
        fBins = int(2**np.ceil(np.log2(2*nBins)))
    else:
        fBins = nBins

    V = np.fft.fftfreq(fBins,d=dBin)
    N = V*fBins
    
    ramp_filter = np.zeros(fBins)
    ramp_filter[0] = 1./(4.0*dBin**2)
    ramp_filter[1::2] = -1.0/(dBin*np.pi*N[1::2])**2
    
    #return (V, ramp_filter)
    
    if return_freq:
        return (V, np.real(np.fft.fft(ramp_filter)))
    else:
        return np.real(np.fft.fft(ramp_filter))


def filter_sino(Sino, dCol=1.0, filter_name='ramp', filter_params=None):

    if filter_name == None:
        return Sino
    
    nDims = Sino.ndim
    
    if nDims == 2:
        nRows, nCols = Sino.shape
    elif nDims == 3:
        nProjs, nRows, nCols = Sino.shape
    else:
        raise ValueError('Sino must have (nRows,nCols) or (nProjs,nRows,nCols) shape')
           
        
    z = int(2**np.ceil(np.log2(2*nCols)))


    #Create vector of the filter (n = zpbins)
    if filter_params == None:
        cutoff = 0.5
            
    Filter1D = ramp(nCols, dBin=dCol, cutoff=cutoff)
 
    if filter_name == 'gaus':
        Filter1D *= GaussianFilter(nCols, sigma=0.68)

    if filter_name == 'hann':
        Filter1D *= 0.5+0.5*np.cos(np.pi*np.arange(z)/(z*cutoff))
        
    #Returns the product of the ramp filter above and a Hanning window.
    #fan_ramp = fan_ramp(cutoff,bins, orig_bins, bin_size, equispaced=equispaced)
    #absic = shift(findgen(bins)-bins/2.+1.,bins/2+1)
    #hann = 0.5+0.5*cos(!PI*absic/(bins*cutoff))
    #fil = fan_ramp*hann
             
    return np.fft.irfft(dCol*Filter1D* \
        np.fft.fft(Sino, axis=nDims-1, n=z), n=z, axis=nDims-1)[:,:,:nCols]


def ramp_old(nBins, cutoff=0.5, dBin=1.0, equispaced=False):
    z = int(2**np.ceil(np.log2(2*nBins)))

    Vs = np.fft.fftfreq(z)
    Vs2 = np.fft.fftfreq(z,d=1.0/z)
    
    fil = np.zeros([z])
    fil[0] = 1./(8.0*dBin**2)
    fil[1::2] = -0.5/(dBin*np.pi*Vs2[1::2])**2
    fil = np.abs(np.fft.fft(np.fft.fftshift(fil), n=z))
    print(dBin*Vs2[1::2])
    
    fil[np.abs(Vs) > cutoff] = 0.0

    return fil





def fan_ramp(nBins, dBin=1.0, cutoff=0.5, equispaced=False):
    
    #Determine the length of the zero-padded sequence.
    #This should be smallest power of 2 that is greater than or equal to (2*nbins-1)
    #to eliminate the interperiod interference artifacts inherent to periodic convolution (2*nbins-1)
    #and be the smallest power of 2 to reduce the computation in FFT
    zpBins = (2**np.ceil(np.log(2.*nBins-1.)/np.log(2.))).astype(int)

    coords = np.linspace(0,zpBins-1,zpBins) - zpBins/2.0
    freq_absic = np.abs(np.roll(coords/zpBins,(-zpBins//2)))

    if(equispaced == False):
        coords  = np.sin(dBin*coords)
    else:
        coords = dBin*coords

    #Calculates the frequency position 
    w = np.arange(1,nBins,2)

    #Creates the filter kernel in frequency space
    fil = np.zeros([zpBins])
    fil[zpBins//2 + w]       = -0.5*(1.0/(np.pi*coords[zpBins//2 + w]))**2
    fil[zpBins//2 - w[::-1]] = -0.5*(1.0/(np.pi*coords[zpBins//2 - w[::-1]]))**2
    fil[zpBins//2]           = 1./(8.0*dBin**2)
    fil = np.abs(np.fft.fft(fil))

    pos = np.argwhere(freq_absic > cutoff)
    fil[pos] = 0.0

    return fil




def filter_sino_old(Sino, dCol=1.0, filter_name='ramp', filter_params=None):

    if filter_name == None:
        return Sino
    
    nDims = Sino.ndim
    
    if nDims == 2:
        nRows, nCols = Sino.shape
    elif nDims == 3:
        nProjs, nRows, nCols = Sino.shape
    else:
        raise ValueError('Sino must have (nRows,nCols) or (nProjs,nRows,nCols) shape')
           
        
    z = int(2**np.ceil(np.log2(2*nCols)))


    #Create vector of the filter (n = zpbins)
    if filter_params == None:
        cutoff = 0.5
            
    Filter1D = fan_ramp(nCols, dBin=dCol, cutoff=cutoff)
 
    if filter_name == 'gaus':
        Filter1D *= GaussianFilter(nCols, sigma=0.68)

    if filter_name == 'hann':
        Filter1D *= 0.5+0.5*np.cos(np.pi*np.arange(z)/(z*cutoff))
        
    #Returns the product of the ramp filter above and a Hanning window.
    #fan_ramp = fan_ramp(cutoff,bins, orig_bins, bin_size, equispaced=equispaced)
    #absic = shift(findgen(bins)-bins/2.+1.,bins/2+1)
    #hann = 0.5+0.5*cos(!PI*absic/(bins*cutoff))
    #fil = fan_ramp*hann
             
    #Filter the sinogram in the column dimension
    return np.fft.irfft(dCol*Filter1D* \
        np.fft.fft(Sino, axis=nDims-1, n=z), n=z, axis=nDims-1)[:,:nCols]
        

def rampFT(nBins, dBin=1.0, cutoff=0.5, equispaced=False):
    
    #Determine the length of the zero-padded sequence.
    #This should be smallest power of 2 that is greater than or equal to (2*nbins-1)
    #to eliminate the interperiod interference artifacts inherent to periodic convolution (2*nbins-1)
    #and be the smallest power of 2 to reduce the computation in FFT
    zpBins = (2**np.ceil(np.log(2.*nBins-1.)/np.log(2.))).astype(int)

    coords = np.linspace(0,zpBins-1,zpBins) - zpBins/2.0
    freq_absic = np.abs(np.roll(coords/zpBins,(-zpBins/2)))

    if(equispaced == False):
        coords  = np.sin(dBin*coords)
    else:
        coords = dBin*coords

    #Calculates the frequency position 
    w = range(1,nBins,2)

    #Creates the filter kernel in frequency space
    fil = np.zeros([zpBins])
    fil[zpBins/2 + w]       = -0.5*(1.0/(np.pi*coords[zpBins/2 + w]))**2
    fil[zpBins/2 - w[::-1]] = -0.5*(1.0/(np.pi*coords[zpBins/2 - w[::-1]]))**2
    fil[zpBins/2]           = 1./(8.0*dBin**2)

    return fil
    
def GaussianFilter(nBins, sigma=0.68):
    z = 2**np.ceil(np.log2(2*nBins-1)).astype(int)
    
    fil = np.zeros([z])
    if sigma == 0:
        fil[0] = 1.0
    else:
        fil[0:z/2] = np.exp(-1.0*(.5/sigma**2)*np.arange(z/2)**2)
    
    fil[z/2+1:] = fil[1:z/2][::-1]
    fil /= fil.sum()
    return np.fft.fft(fil)

def sci_filter(radon_image):
    
    img_shape = radon_image.shape[1]
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Apply filter in Fourier domain
    fourier_filter = _get_fourier_filter(projection_size_padded, filter)
    projection = np.fft.fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(np.fft.ifft(projection, axis=0)[:img_shape, :])

    return radon_filtered



def _get_fourier_filter(size, filter_name):
    """Construct the Fourier filter.

    This computation lessens artifacts and removes a small bias as
    explained in [1], Chap 3. Equation 61.

    Parameters
    ----------
    size: int
        filter size. Must be even.
    filter_name: str
        Filter used in frequency domain filtering. Filters available:
        ramp, shepp-logan, cosine, hamming, hann. Assign None to use
        no filter.

    Returns
    -------
    fourier_filter: ndarray
        The computed Fourier filter.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.

    """
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=np.int),
                        np.arange(size / 2 - 1, 0, -2, dtype=np.int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # Computing the ramp filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in [1], Chap 3. Equation 61
    fourier_filter = 2 * np.real(np.fft.fft(f))         # ramp filter
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * np.fft.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = np.fft.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= np.fft.fftshift(np.hamming(size))
    elif filter_name == "hann":
        fourier_filter *= np.fft.fftshift(np.hanning(size))
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter[:, np.newaxis]

