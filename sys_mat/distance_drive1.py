import numpy as np

def distance_driven_fp(image, angles, det_count, pixel_size=1.0, det_spacing=1.0):
    """
    Distance-driven forward projection (2D, parallel-beam, no rotation)

    Parameters
    ----------
    image : ndarray (ny, nx)
        Input image
    angles : ndarray
        Projection angles (radians)
    det_count : int
        Number of detector bins
    pixel_size : float
        Pixel size
    det_spacing : float
        Detector spacing

    Returns
    -------
    sinogram : ndarray (n_angles, det_count)
    """

    ny, nx = image.shape
    sinogram = np.zeros((len(angles), det_count))

    # Detector bin edges
    det_edges = (
        np.arange(det_count + 1) - det_count / 2
    ) * det_spacing

    # Pixel edges
    x_edges = (np.arange(nx + 1) - nx / 2) * pixel_size
    y_edges = (np.arange(ny + 1) - ny / 2) * pixel_size

    for ia, theta in enumerate(angles):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Choose dominant axis
        if abs(cos_t) > abs(sin_t):
            # Integrate along y, project x
            for ix in range(nx):
                x0 = x_edges[ix]
                x1 = x_edges[ix + 1]

                # Project pixel x-edges onto detector
                s0 = x0 * cos_t
                s1 = x1 * cos_t
                s_min, s_max = sorted((s0, s1))

                for idet in range(det_count):
                    d0 = det_edges[idet]
                    d1 = det_edges[idet + 1]

                    overlap = max(0.0, min(s_max, d1) - max(s_min, d0))
                    if overlap > 0:
                        sinogram[ia, idet] += overlap * np.sum(image[:, ix])

        else:
            # Integrate along x, project y
            for iy in range(ny):
                y0 = y_edges[iy]
                y1 = y_edges[iy + 1]

                # Project pixel y-edges onto detector
                s0 = y0 * sin_t
                s1 = y1 * sin_t
                s_min, s_max = sorted((s0, s1))

                for idet in range(det_count):
                    d0 = det_edges[idet]
                    d1 = det_edges[idet + 1]

                    overlap = max(0.0, min(s_max, d1) - max(s_min, d0))
                    if overlap > 0:
                        sinogram[ia, idet] += overlap * np.sum(image[iy, :])

    return sinogram
