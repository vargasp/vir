#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:38:59 2020

@author: vargasp
"""

import numpy as np

from scipy.interpolate import interp1d
from scipy.constants import golden_ratio
from warnings import warn
from functools import partial
from numpy.fft import fft, ifft



__all__ = ['radon', 'order_angles_golden_ratio', 'iradon', 'iradon_sart']


def radon(image, theta=None, circle=True, *, preserve_range=None):
    """
    Calculates the radon transform of an image given specified
    projection angles.

    Parameters
    ----------
    image : array_like
        Input image. The rotation axis will be located in the pixel with
        indices ``(image.shape[0] // 2, image.shape[1] // 2)``.
    theta : array_like, optional
        Projection angles (in degrees). If `None`, the value is set to
        np.arange(180).
    circle : boolean, optional
        Assume image is zero outside the inscribed circle, making the
        width of each projection (the first dimension of the sinogram)
        equal to ``min(image.shape)``.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html

    Returns
    -------
    radon_image : ndarray
        Radon transform (sinogram).  The tomography rotation axis will lie
        at the pixel index ``radon_image.shape[0] // 2`` along the 0th
        dimension of ``radon_image``.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
           the Discrete Radon Transform With Some Applications", Proceedings of
           the Fourth IEEE Region 10 International Conference, TENCON '89, 1989

    Notes
    -----
    Based on code of Justin K. Romberg
    (https://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html)

    """
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        theta = np.arange(180)

    if preserve_range is None and np.issubdtype(image.dtype, np.integer):
        warn('Image dtype is not float. By default radon will assume '
             'you want to preserve the range of your image '
             '(preserve_range=True). In scikit-image 0.18 this behavior will '
             'change to preserve_range=False. To avoid this warning, '
             'explicitly specify the preserve_range parameter.',
             stacklevel=2)
        preserve_range = True

    image = convert_to_float(image, preserve_range)

    if circle:
        shape_min = min(image.shape)
        radius = shape_min // 2
        img_shape = np.array(image.shape)
        coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]])
        dist = ((coords - img_shape // 2) ** 2).sum(0)
        outside_reconstruction_circle = dist > radius ** 2
        if np.any(image[outside_reconstruction_circle]):
            warn('Radon transform: image must be zero outside the '
                 'reconstruction circle')
        # Crop image to make it square
        slices = tuple(slice(int(np.ceil(excess / 2)),
                             int(np.ceil(excess / 2) + shape_min))
                       if excess > 0 else slice(None)
                       for excess in (img_shape - shape_min))
        padded_image = image[slices]
    else:
        diagonal = np.sqrt(2) * max(image.shape)
        pad = [int(np.ceil(diagonal - s)) for s in image.shape]
        new_center = [(s + p) // 2 for s, p in zip(image.shape, pad)]
        old_center = [s // 2 for s in image.shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = np.pad(image, pad_width, mode='constant',
                              constant_values=0)

    # padded_image is always square
    if padded_image.shape[0] != padded_image.shape[1]:
        raise ValueError('padded_image must be a square')
    center = padded_image.shape[0] // 2
    radon_image = np.zeros((padded_image.shape[0], len(theta)))

    for i, angle in enumerate(np.deg2rad(theta)):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                      [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                      [0, 0, 1]])
        rotated = warp(padded_image, R, clip=False)
        radon_image[:, i] = rotated.sum(0)
    return radon_image


def _sinogram_circle_to_square(sinogram):
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
    pad = diagonal - sinogram.shape[0]
    old_center = sinogram.shape[0] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = ((pad_before, pad - pad_before), (0, 0))
    return np.pad(sinogram, pad_width, mode='constant', constant_values=0)


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
    fourier_filter = 2 * np.real(fft(f))         # ramp filter
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * fftmodule.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = fftmodule.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= fftmodule.fftshift(np.hamming(size))
    elif filter_name == "hann":
        fourier_filter *= fftmodule.fftshift(np.hanning(size))
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter[:, np.newaxis]


def iradon(radon_image, theta=None, output_size=None,
           filter="ramp", interpolation="linear", circle=True):
    """Inverse radon transform.

    Reconstruct an image from the radon transform, using the filtered
    back projection algorithm.

    Parameters
    ----------
    radon_image : array_like, dtype=float
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different
        angle. The tomography rotation axis should lie at the pixel
        index ``radon_image.shape[0] // 2`` along the 0th dimension of
        ``radon_image``.
    theta : array_like, dtype=float, optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    output_size : int, optional
        Number of rows and columns in the reconstruction.
    filter : str, optional
        Filter used in frequency domain filtering. Ramp filter used by default.
        Filters available: ramp, shepp-logan, cosine, hamming, hann.
        Assign None to use no filter.
    interpolation : str, optional
        Interpolation method used in reconstruction. Methods available:
        'linear', 'nearest', and 'cubic' ('cubic' is slow).
    circle : boolean, optional
        Assume the reconstructed image is zero outside the inscribed circle.
        Also changes the default output_size to match the behaviour of
        ``radon`` called with ``circle=True``.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image. The rotation axis will be located in the pixel
        with indices
        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
           the Discrete Radon Transform With Some Applications", Proceedings of
           the Fourth IEEE Region 10 International Conference, TENCON '89, 1989

    Notes
    -----
    It applies the Fourier slice theorem to reconstruct an image by
    multiplying the frequency domain of the filter with the FFT of the
    projection data. This algorithm is called filtered back projection.

    """
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')

    if theta is None:
        theta = np.linspace(0, 180, radon_image.shape[1], endpoint=False)

    angles_count = len(theta)
    if angles_count != radon_image.shape[1]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")

    interpolation_types = ('linear', 'nearest', 'cubic')
    if interpolation not in interpolation_types:
        raise ValueError("Unknown interpolation: %s" % interpolation)

    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter not in filter_types:
        raise ValueError("Unknown filter: %s" % filter)

    img_shape = radon_image.shape[0]
    if output_size is None:
        # If output size not specified, estimate from input radon image
        if circle:
            output_size = img_shape
        else:
            output_size = int(np.floor(np.sqrt((img_shape) ** 2 / 2.0)))

    if circle:
        radon_image = _sinogram_circle_to_square(radon_image)
        img_shape = radon_image.shape[0]

    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Apply filter in Fourier domain
    fourier_filter = _get_fourier_filter(projection_size_padded, filter)
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0)[:img_shape, :])

    # Reconstruct image by interpolation
    reconstructed = np.zeros((output_size, output_size))
    radius = output_size // 2
    xpr, ypr = np.mgrid[:output_size, :output_size] - radius
    x = np.arange(img_shape) - img_shape // 2

    for col, angle in zip(radon_filtered.T, np.deg2rad(theta)):
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        if interpolation == 'linear':
            interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
        else:
            interpolant = interp1d(x, col, kind=interpolation,
                                   bounds_error=False, fill_value=0)
        reconstructed += interpolant(t)

    if circle:
        out_reconstruction_circle = (xpr ** 2 + ypr ** 2) > radius ** 2
        reconstructed[out_reconstruction_circle] = 0.

    return reconstructed * np.pi / (2 * angles_count)


def order_angles_golden_ratio(theta):
    """Order angles to reduce the amount of correlated information in
    subsequent projections.

    Parameters
    ----------
    theta : 1D array of floats
        Projection angles in degrees. Duplicate angles are not allowed.

    Returns
    -------
    indices_generator : generator yielding unsigned integers
        The returned generator yields indices into ``theta`` such that
        ``theta[indices]`` gives the approximate golden ratio ordering
        of the projections. In total, ``len(theta)`` indices are yielded.
        All non-negative integers < ``len(theta)`` are yielded exactly once.

    Notes
    -----
    The method used here is that of the golden ratio introduced
    by T. Kohler.

    References
    ----------
    .. [1] Kohler, T. "A projection access scheme for iterative
           reconstruction based on the golden section." Nuclear Science
           Symposium Conference Record, 2004 IEEE. Vol. 6. IEEE, 2004.
    .. [2] Winkelmann, Stefanie, et al. "An optimal radial profile order
           based on the Golden Ratio for time-resolved MRI."
           Medical Imaging, IEEE Transactions on 26.1 (2007): 68-76.

    """
    interval = 180

    remaining_indices = list(np.argsort(theta))   # indices into theta
    # yield an arbitrary angle to start things off
    angle = theta[remaining_indices[0]]
    yield remaining_indices.pop(0)
    # determine subsequent angles using the golden ratio method
    angle_increment = interval / golden_ratio ** 2
    while remaining_indices:
        remaining_angles = theta[remaining_indices]
        angle = (angle + angle_increment) % interval
        index_above = np.searchsorted(remaining_angles, angle)
        index_below = index_above - 1
        index_above %= len(remaining_indices)

        diff_below = abs(angle - remaining_angles[index_below])
        distance_below = min(diff_below % interval, diff_below % -interval)

        diff_above = abs(angle - remaining_angles[index_above])
        distance_above = min(diff_above % interval, diff_above % -interval)

        if distance_below < distance_above:
            yield remaining_indices.pop(index_below)
        else:
            yield remaining_indices.pop(index_above)


def iradon_sart(radon_image, theta=None, image=None, projection_shifts=None,
                clip=None, relaxation=0.15):
    """Inverse radon transform.

    Reconstruct an image from the radon transform, using a single iteration of
    the Simultaneous Algebraic Reconstruction Technique (SART) algorithm.

    Parameters
    ----------
    radon_image : 2D array, dtype=float
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different angle. The
        tomography rotation axis should lie at the pixel index
        ``radon_image.shape[0] // 2`` along the 0th dimension of
        ``radon_image``.
    theta : 1D array, dtype=float, optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    image : 2D array, dtype=float, optional
        Image containing an initial reconstruction estimate. Shape of this
        array should be ``(radon_image.shape[0], radon_image.shape[0])``. The
        default is an array of zeros.
    projection_shifts : 1D array, dtype=float, optional
        Shift the projections contained in ``radon_image`` (the sinogram) by
        this many pixels before reconstructing the image. The i'th value
        defines the shift of the i'th column of ``radon_image``.
    clip : length-2 sequence of floats, optional
        Force all values in the reconstructed tomogram to lie in the range
        ``[clip[0], clip[1]]``
    relaxation : float, optional
        Relaxation parameter for the update step. A higher value can
        improve the convergence rate, but one runs the risk of instabilities.
        Values close to or higher than 1 are not recommended.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image. The rotation axis will be located in the pixel
        with indices
        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.

    Notes
    -----
    Algebraic Reconstruction Techniques are based on formulating the tomography
    reconstruction problem as a set of linear equations. Along each ray,
    the projected value is the sum of all the values of the cross section along
    the ray. A typical feature of SART (and a few other variants of algebraic
    techniques) is that it samples the cross section at equidistant points
    along the ray, using linear interpolation between the pixel values of the
    cross section. The resulting set of linear equations are then solved using
    a slightly modified Kaczmarz method.

    When using SART, a single iteration is usually sufficient to obtain a good
    reconstruction. Further iterations will tend to enhance high-frequency
    information, but will also often increase the noise.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] AH Andersen, AC Kak, "Simultaneous algebraic reconstruction
           technique (SART): a superior implementation of the ART algorithm",
           Ultrasonic Imaging 6 pp 81--94 (1984)
    .. [3] S Kaczmarz, "Angenäherte auflösung von systemen linearer
           gleichungen", Bulletin International de l’Academie Polonaise des
           Sciences et des Lettres 35 pp 355--357 (1937)
    .. [4] Kohler, T. "A projection access scheme for iterative
           reconstruction based on the golden section." Nuclear Science
           Symposium Conference Record, 2004 IEEE. Vol. 6. IEEE, 2004.
    .. [5] Kaczmarz' method, Wikipedia,
           https://en.wikipedia.org/wiki/Kaczmarz_method

    """
    if radon_image.ndim != 2:
        raise ValueError('radon_image must be two dimensional')
    reconstructed_shape = (radon_image.shape[0], radon_image.shape[0])
    if theta is None:
        theta = np.linspace(0, 180, radon_image.shape[1], endpoint=False)
    elif theta.shape != (radon_image.shape[1],):
        raise ValueError('Shape of theta (%s) does not match the '
                         'number of projections (%d)'
                         % (projection_shifts.shape, radon_image.shape[1]))
    if image is None:
        image = np.zeros(reconstructed_shape, dtype=np.float)
    elif image.shape != reconstructed_shape:
        raise ValueError('Shape of image (%s) does not match first dimension '
                         'of radon_image (%s)'
                         % (image.shape, reconstructed_shape))
    if projection_shifts is None:
        projection_shifts = np.zeros((radon_image.shape[1],), dtype=np.float)
    elif projection_shifts.shape != (radon_image.shape[1],):
        raise ValueError('Shape of projection_shifts (%s) does not match the '
                         'number of projections (%d)'
                         % (projection_shifts.shape, radon_image.shape[1]))
    if clip is not None:
        if len(clip) != 2:
            raise ValueError('clip must be a length-2 sequence')
        clip = (float(clip[0]), float(clip[1]))
    relaxation = float(relaxation)

    for angle_index in order_angles_golden_ratio(theta):
        image_update = sart_projection_update(image, theta[angle_index],
                                              radon_image[:, angle_index],
                                              projection_shifts[angle_index])
        image += relaxation * image_update
        if clip is not None:
            image = np.clip(image, clip[0], clip[1])
    return image



def sci_filter(radon_image, theta=None, output_size=None,
           filter="ramp", interpolation="linear", circle=True):
    """Inverse radon transform.

    Reconstruct an image from the radon transform, using the filtered
    back projection algorithm.

    Parameters
    ----------
    radon_image : array_like, dtype=float
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different
        angle. The tomography rotation axis should lie at the pixel
        index ``radon_image.shape[0] // 2`` along the 0th dimension of
        ``radon_image``.
    theta : array_like, dtype=float, optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    output_size : int, optional
        Number of rows and columns in the reconstruction.
    filter : str, optional
        Filter used in frequency domain filtering. Ramp filter used by default.
        Filters available: ramp, shepp-logan, cosine, hamming, hann.
        Assign None to use no filter.
    interpolation : str, optional
        Interpolation method used in reconstruction. Methods available:
        'linear', 'nearest', and 'cubic' ('cubic' is slow).
    circle : boolean, optional
        Assume the reconstructed image is zero outside the inscribed circle.
        Also changes the default output_size to match the behaviour of
        ``radon`` called with ``circle=True``.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image. The rotation axis will be located in the pixel
        with indices
        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
           the Discrete Radon Transform With Some Applications", Proceedings of
           the Fourth IEEE Region 10 International Conference, TENCON '89, 1989

    Notes
    -----
    It applies the Fourier slice theorem to reconstruct an image by
    multiplying the frequency domain of the filter with the FFT of the
    projection data. This algorithm is called filtered back projection.

    """
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')

    if theta is None:
        theta = np.linspace(0, 180, radon_image.shape[1], endpoint=False)

    angles_count = len(theta)
    if angles_count != radon_image.shape[1]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")

    interpolation_types = ('linear', 'nearest', 'cubic')
    if interpolation not in interpolation_types:
        raise ValueError("Unknown interpolation: %s" % interpolation)

    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter not in filter_types:
        raise ValueError("Unknown filter: %s" % filter)

    img_shape = radon_image.shape[0]
    if output_size is None:
        # If output size not specified, estimate from input radon image
        if circle:
            output_size = img_shape
        else:
            output_size = int(np.floor(np.sqrt((img_shape) ** 2 / 2.0)))

    if circle:
        radon_image = _sinogram_circle_to_square(radon_image)
        img_shape = radon_image.shape[0]

    print(img_shape,output_size)
    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Apply filter in Fourier domain
    fourier_filter = _get_fourier_filter(projection_size_padded, filter)
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0)[:img_shape, :])
    
    
    return radon_filtered


