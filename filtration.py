import numpy as np


def ramp_filter(nu, du=1.0, cutoff=0.5, zero_pad=True,
                half_spectrum=True, real_space=True,
                return_freq=False):
    """
    Construct a ramp (Ram-Lak) filter for filtered backprojection (FBP); The
    ramp filter is H(f) = |f|.

    The filter can be generated either:
    - in real space (via the Kak & Slaney convolution kernel), or
    - directly in the frequency domain (ASTRA-style).

    Parameters
    ----------
    nu : int
        Number of detector samples (projection width) [unitless]

    du : float, optional
        Detector spacing (distance between samples) [length].
        This defines the physical sampling of the projection and determines
        the Nyquist frequency: f_nyq = 1 / (2*du).

    cutoff : float, optional
        Cutoff frequency expressed as a fraction of the sampling frequency f_s,
        where f_s = 1 / du.
        
        NOTE:
        - cutoff = 0.5 corresponds to the Nyquist frequency
        - cutoff = 1.0 corresponds to the full sampling frequency (not typical)
        - typical range: (0, 0.5]

    zero_pad : bool, optional
        If True, zero-pad the detector axis to the next power of 2 ≥ 2*nu.
        This reduces circular convolution artifacts and improves FFT efficiency.

    half_spectrum : bool, optional
        If True, return only the non-negative frequency half-spectrum
        (compatible with rFFT). Otherwise return full FFT spectrum.

    real_space : bool, optional
        If True, construct the filter via the spatial-domain kernel and FFT it.
        If False, construct the filter directly in frequency space.

    Returns
    -------
    H : ndarray
        Ramp filter in frequency domain.
        Shape:
            - (n//2 + 1,) if half_spectrum=True
            - (n,) otherwise
        Units: [1 / length]
        
    f : frequencies (physical units: cycles per length)
    """

    # Determine FFT length (zero-padding improves frequency sampling)
    if zero_pad:
        nfu = int(2**np.ceil(np.log2(2 * nu)))
    else:
        nfu = nu


    # Frequency axis (physical units: cycles per length)
    if half_spectrum:
        f = np.fft.rfftfreq(nfu, d=du)
    else:
        f = np.fft.fftfreq(nfu, d=du)

    # Sampling and Nyquist frequencies
    f_s = 1.0 / du          # sampling frequency [1 / length]

    # --- REAL-SPACE CONSTRUCTION ---
    if real_space:
        # Construct integer index k in FFT ordering:
        # [0, 1, 2, ..., n/2, -n/2+1, ..., -1]
        #
        # Using fftfreq with d=1/n produces integer-valued bins.
        # This is a convenient way to match FFT indexing exactly.
        k = np.fft.fftfreq(nfu, d=1.0 / nfu)

        # Initialize spatial-domain kernel h[k]
        h = np.zeros(nfu)

        # Central value (k = 0)
        # Units: [1 / length^2]
        h[0] = 1.0 / (4.0 * du**2)

        # Odd indices only (k = ±1, ±3, ...)
        # Formula: h[k] = -1 / (π^2 * du^2 * k^2)
        h[1::2] = -1.0 / (np.pi**2 * du**2 * k[1::2]**2)

        # Transform to frequency domain
        if half_spectrum:
            H = du * np.real(np.fft.rfft(h)) # scale by du (integral approximation)
        else:
            H = du * np.real(np.fft.fft(h))

    # --- FREQUENCY-DOMAIN CONSTRUCTION ---
    else:
        # Ramp filter: H(f) = |f|
        H = np.abs(f)

    # Apply cutoff (expressed as fraction of sampling frequency)
    # cutoff * f_s → actual cutoff frequency
    H[np.abs(f) > cutoff * f_s] = 0.0

    if return_freq:
        return H, f
    else:
        return H


def window_filter(H, f, du=1.0, filter_type="ramp", cutoff=0.5, sigma=0.5):
    """
    Apply a window function to a ramp filter in the frequency domain.

    This function modifies a precomputed ramp filter H(f) by applying a
    frequency window W(f), producing a band-limited filter used in
    filtered backprojection (FBP).

    All window functions are defined in terms of *physical frequency* and
    are properly normalized with respect to the cutoff frequency, ensuring
    consistent behavior when the detector spacing `du` changes.

    Parameters
    ----------
    H : ndarray
        Input ramp filter evaluated at frequencies `f`.
        Typically H(f) = |f| or a discretized equivalent.
        Units: [1 / length]

    f : ndarray
        Frequency axis corresponding to H.
        Must be in physical units: cycles per unit length (e.g., 1/mm).

    du : float, optional
        Detector spacing (sample spacing in projection domain).
        Defines the sampling frequency: f_s = 1 / du.

    filter_type : str, optional
        Type of window to apply. Options:
        - "ramp" or "ram-lak" : no window (pure ramp filter)
        - "hann"              : Hann window
        - "hamming"           : Hamming window
        - "blackman"          : Blackman window
        - "shepp-logan"       : Shepp-Logan (sinc) window
        - "cosine"            : Cosine window
        - "gaussian"          : Gaussian window (no hard cutoff)

    cutoff : float, optional
        Cutoff frequency as a fraction of the sampling frequency f_s.
        The physical cutoff is f_c = cutoff * f_s.
        Typical range: (0, 0.5].

    sigma : float, optional
        Standard deviation for the Gaussian window in *physical frequency units*.
        Only used if filter_type == "gaussian".

    Returns
    -------
    H_windowed : ndarray
        The windowed filter H(f) * W(f), same shape as input H.

    Notes
    -----
    - All windows except Gaussian are band-limited to |f| <= f_c.
    - Window functions are defined in terms of normalized frequency:
          f_norm = f / f_c
    - The Shepp–Logan window uses:
          W(f) = sinc(f / (2 f_c))
      and does NOT go to zero at the cutoff.
    - Gaussian window does not enforce a hard cutoff.
    """

    # Sampling frequency (cycles per unit length)
    f_s = 1.0 / du

    # Physical cutoff frequency
    f_c = cutoff * f_s

    # Normalize frequency to cutoff (dimensionless)
    f_norm = f / f_c

    # Mask of frequencies within cutoff band
    valid = np.abs(f) <= f_c

    # Initialize window
    W = np.zeros_like(f)

    # --- No window (pure ramp) ---
    if filter_type == "ram-lak" or filter_type == "ramp":
        return H

    # --- Hann window ---
    # W(f) = 0.5 + 0.5 cos(pi f / f_c)
    elif filter_type == "hann":
        W[valid] = 0.5 + 0.5 * np.cos(np.pi * f_norm[valid])

    # --- Hamming window ---
    # W(f) = 0.54 + 0.46 cos(pi f / f_c)
    elif filter_type == "hamming":
        W[valid] = 0.54 + 0.46 * np.cos(np.pi * f_norm[valid])

    # --- Blackman window ---
    # W(f) = 0.42 + 0.5 cos(pi f / f_c) + 0.08 cos(2π f / f_c)
    elif filter_type == "blackman":
        W[valid] = (
            0.42
            + 0.5 * np.cos(np.pi * f_norm[valid])
            + 0.08 * np.cos(2 * np.pi * f_norm[valid])
        )

    # --- Shepp-Logan window ---
    # W(f) = sinc(f / (2 f_c)) = sin(pi f / (2 f_c)) / (pi f / (2 f_c))
    # Does NOT go to zero at f = f_c
    elif filter_type == "shepp-logan":
        W[0] = 1.0  # limit as f -> 0
        m = valid & (f != 0)
        W[m] = np.sin(np.pi * f[m] / (2 * f_c)) / (np.pi * f[m] / (2 * f_c))

    # --- Cosine window ---
    # W(f) = cos(pi f / (2 f_c))
    elif filter_type == "cosine":
        W[valid] = np.cos(np.pi * f_norm[valid] / 2.0)

    # --- Gaussian window ---
    # W(f) = exp(-f^2 / (2 sigma^2))
    # Note: no hard cutoff applied here
    elif filter_type == "gaussian":
        W = np.exp(-0.5 * (f / sigma) ** 2)

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Apply window to ramp filter
    return H * W
    

def filter_sino(sino, du=1.0, filter_name='ramp', filter_params=None):
    """
    Apply a 1D frequency-domain filter to the projections (sinogram) along the last axis.

    This implements filtered backprojection (FBP) style filtering using a ramp
    filter, optionally combined with a window (Hanning or Gaussian). The filtering
    is done in the Fourier domain using `np.fft.rfft`/`irfft` for efficiency.

    Parameters
    ----------
    sino : ndarray
        The input sinogram array. Filtering is applied along the last axis
        (detector columns).
    du : float, optional
        Detector sampling interval (physical spacing between detector elements).
        Default is 1.0.
    filter_name : str or None, optional
        Type of windowing to apply to the ramp filter:
        - 'ramp' (default) : pure ramp filter
        - 'hann'           : ramp multiplied by Hanning window
        - 'gaus'           : ramp multiplied by Gaussian window
        - None             : no filtering applied (returns original sinogram)
    filter_params : dict or None, optional
        Dictionary of filter-specific parameters. Currently not used; included
        for future extensibility (e.g., cutoff frequency, Gaussian sigma).

    Returns
    -------
    filtered : ndarray
        The filtered sinogram array, same shape as `sino`.

    Notes
    -----
    - The function zero-pads the detector axis to the next power of two >= 2*nu
      to reduce periodic convolution artifacts in the FFT.
    - Filtering is performed in-place in the Fourier domain to minimize temporary
      arrays and memory usage.
    - The `cutoff` parameter (currently set to 0.5 if `filter_params` is None)
      represents a fraction of the Nyquist frequency and is applied when using
      windowed filters ('hann' or 'gaus').
    - The ramp filter is computed using the `ramp()` function, which builds
      a symmetric filter in the spatial domain and then Fourier transforms it.
    - The continuous inverse Radon transform includes a factor 1/2.
      Discrete implementations typically absorb this into the filter,
      giving an effective ramp of 2|f|. We apply the factor here so the
      reconstruction matches standard implementations
    """

    if filter_name == None:
        return sino
    
    nu = sino.shape[-1]
               
    # FFT size
    z = int(2**np.ceil(np.log2(2*nu)))

    #Create vector of the filter (n = zpbins)
    if filter_params == None:
        cutoff = 0.5
            
    #H = ramp_filter_old(nu, du=du, cutoff=cutoff, zero_pad=True)
 
    H = ramp_filter(nu, du=du, cutoff=cutoff, zero_pad=True,
                    half_spectrum=True, real_space=True)
 
    
    if filter_name == 'gaus':
        H *= gaussian_filter(z, sigma=0.68)

    elif filter_name == 'hann':
        H *= hann_filter(z, cutoff=0.5)
    
                 
    # Compute rFFT of sino directly (half-spectrum)
    fft_sino = np.fft.rfft(sino, axis=-1, n=z)
        
    # FBP normalization (see note)
    fft_sino *= 2.0 * H
    
    # Inverse rFFT (half-spectrum)
    filtered = np.fft.irfft(fft_sino, n=z, axis=-1) 

    # Crop to original size
    return filtered[..., :nu]
    


def astra_filter(n, du=1.0, filter_type="ram-lak", cutoff=1.0, param=None):
    """
    Create ASTRA-style FBP filters in the frequency domain (rFFT form).

    Parameters
    ----------
    n : int
        FFT length (after zero-padding).
    du : float, optional
        Detector spacing.
    filter_type : str
        One of:
        - 'ram-lak'      : pure ramp
        - 'hann'         : ramp * Hann window
        - 'shepp-logan'  : ramp * sinc window
        - 'cosine'       : ramp * cosine window
        - 'gaussian'     : ramp * Gaussian window
    cutoff : float, optional
        Fraction of Nyquist (0 < cutoff ≤ 1). ASTRA uses this as scaling.
    param : float or None
        Optional parameter (used for Gaussian sigma, etc.)

    Returns
    -------
    H : ndarray
        Real-valued frequency response (length n//2 + 1 for rFFT)
    """

    # rFFT frequencies (cycles per unit length)
    freq = np.fft.rfftfreq(n, d=du)

    # Nyquist frequency
    f_nyq = 1.0 / (2.0 * du)

    # Normalize frequency to [0, 1] relative to cutoff
    f = freq / (cutoff * f_nyq)

    # --- Ramp (Ram-Lak) ---
    H = np.abs(freq)

    # --- Window functions ---
    if filter_type == "ram-lak":
        pass

    elif filter_type == "hann":
        # Hann window: 0.5 + 0.5 cos(pi f)
        W = np.zeros_like(f)
        mask = f <= 1
        W[mask] = 0.5 + 0.5 * np.cos(np.pi * f[mask])
        H *= W

    elif filter_type == "shepp-logan":
        # sinc window: sin(pi f) / (pi f)
        W = np.ones_like(f)
        mask = f > 0
        W[mask] = np.sin(np.pi * f[mask]) / (np.pi * f[mask])
        W[f > 1] = 0
        H *= W

    elif filter_type == "cosine":
        W = np.zeros_like(f)
        mask = f <= 1
        W[mask] = np.cos(np.pi * f[mask] / 2)
        H *= W

    elif filter_type == "gaussian":
        sigma = param if param is not None else 0.5
        W = np.exp(-0.5 * (f / sigma) ** 2)
        W[f > 1] = 0
        H *= W

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Zero out beyond cutoff (safety, though windows already enforce it)
    H[f > 1] = 0.0

    # Scale (ASTRA includes factor ~2 for ramp symmetry)
    H *= 2.0

    return H




def ramp_filter_old(nu, du=1.0, cutoff=0.5, zero_pad=True):
    """
    Computers a ramp filter based on AC Kak, M Slaney, "Principles of
    Computerized Tomographic Imaging", IEEE Press 1988.
           
    Parameters
    ----------
    nu : int
        The number of detector columns
    du : float, optional
        The sampling interval between detecros [distance].  Default is 1.0
    cutoff : float
        cutoff is fraction of Nyquist (0 < cutoff ≤ 0.5)
    zero_pad : boolean
        If true zero-pads the filter to the smallest power of 2 that is greater
        or equal to (2*nbins) to eliminate the interperiod interference
        artifacts inherent to periodic convolution. Default is True

    Returns
    -------
    H: (nfu) ndarray
        The computed ramp filter.
    """
    
    if zero_pad:
        nfu = int(2**np.ceil(np.log2(2*nu)))
    else:
        nfu = nu

    v = np.fft.fftfreq(nfu,d=1.0/nfu)
                          
    h = np.zeros(nfu)
    h[0] = 1./(4.0*du**2)
    h[1::2] = -1.0/(du*np.pi*v[1::2])**2

    H = 2 * np.real(np.fft.fft(h))

    if cutoff < 0.5:        
        H[np.abs(v) > cutoff *  nfu / 2] = 0.0
    
    return H





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






