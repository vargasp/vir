import numpy as np
from scipy.optimize import curve_fit


def sine_wave(x, offset, amplitude, phase):
    return np.sin(x + phase) * amplitude + offset


def sinogram(arr, angles, acc_values=0, air_values=10, backlash=False, \
             auto_center=False, center=None, return_cog=False, fluorescence=False, \
             debug=False):



    ncols, nrows = arr.shape

    #If there are an even number of columns in the image throw out the last one
    if ncols % 2 == 0:  ncols -= 1
    #Throw out the acceleration values, convert data to floating point
    arr_output = arr[acc_values:ncols-acc_values, :].astype(float)
    ncols = arr_output.shape[0]

    cog_arr = np.zeros(nrows) # Center-of-gravity array
    linear = np.arange(ncols) / (ncols-1)
    no_air = np.zeros(ncols) + 1e4
    lin2 = np.arange(1,ncols+1)

    for i in range(nrows):
        if air_values > 0:
            air_left = np.sum(arr_output[0:air_values-1,i]) / air_values
            air_right = np.sum(arr_output[ncols-air_values:ncols-1,i]) / air_values
            air = air_left + linear*(air_right-air_left)
        else:
           air = no_air
    
        if not fluorescence:
            #Was arr_output[0,i]
            arr_output[:,i] = -np.log( (arr_output[:,i]/air.clip(1.e-5)))
        
        cog_arr[i] = np.sum(arr_output[:,i] * lin2) / np.sum(arr_output[:,i])

    odds = np.arange(1,nrows,2)
    evens = np.arange(0,nrows,2)

    x = np.deg2rad(angles)
    
    # Initial estimate of rotation axis
    # Initial estimate of amplitude
    # Initial estimate of phase
    a = [(ncols-1)/2., (np.max(cog_arr) - np.min(cog_arr))/2., 0.]                               

    cf_p, pcov = curve_fit(sine_wave, x[odds], cog_arr[odds], p0=a)
    cog_fit = sine_wave(x[odds], cf_p[0], cf_p[1], cf_p[2])
    sigmaa = np.sqrt(np.trace(pcov))

    
    cog_odd = cf_p[0] - 1.
    if debug:
        print(f'Fitted center of gravity for odd rows  = {cog_odd:8.2f} +- {sigmaa:8.2f}')
  
    cf_p, pcov = curve_fit(sine_wave, x[evens], cog_arr[evens], p0=cf_p)
    cog_fit = sine_wave(x[evens], cf_p[0], cf_p[1], cf_p[2])
    sigmaa = np.sqrt(np.trace(pcov))
    
    cog_even = cf_p[0] - 1.
    if debug:
        print(f'Fitted center of gravity for even rows  = {cog_even:8.2f} +- {sigmaa:8.2f}')
   
    """
    if backlash:
        back = cog_even - cog_odd
        if debug:
            print(f'Backlash (even rows shifted right relative to odd rows) = {back:8.2f}')
  
        P = [[back, 0.],[1., 0.]]
        Q = [[0., 1.],[0., 0.]]
        temp = poly_2d(arr_output[:, evens], P, Q, 1)
    
        for i in np.range(0,nrows,2):
            arr_output[0, i] = temp[:, i/2]
            cog[i] = np.sum(arr_output[:,i] * lin2) / np.sum(output[:,i])
    """

    cf_p, pcov = curve_fit(sine_wave, x, cog_arr, p0=cf_p)
    cog_fit = sine_wave(x, cf_p[0], cf_p[1], cf_p[2])
    sigmaa = np.sqrt(np.trace(pcov))
    
    
    cog_mean = cf_p[0] - 1.

    error_before = cog_mean - (ncols-1)/2.


    shift_amount = 0.
    if debug:
        print(f'Fitted center of gravity = {cog_mean:8.2f} +- {sigmaa:8.2f}')        
        print(f'Error before correction (offset from center of image) = {error_before:8.2f}')

    do_shift = False

    if auto_center:
        do_shift = True

    if center != None:
        do_shift = True

    if do_shift:
        if auto_center:
            center = cog_mean
            
        shift_amount = np.round(center - (ncols-1)/2.)

        npad = 2 * np.abs(shift_amount)

        if air_values > 0:
            pad_values = air_values 
        else:
            pad_values = 1

        if shift_amount < 0:
            pad_left = np.zeros([npad, nrows])
            temp = np.sum(arr_output[0:pad_values-1,:],1) / pad_values
    
            for i in range(npad):
                pad_left[i,:] = temp
    
            arr_output = [pad_left, arr_output]
            ncols = arr_output.shape[0]
        
        elif shift_amount > 0:
            pad_right = np.zeros([npad, nrows])           
            temp = np.sum(arr_output[ncols-pad_values:ncols-1,:],1) / pad_values
        
            for i in range(npad):
                pad_right[i,:] = temp
        
            arr_output = [arr_output, pad_right]
            ncols = arr_output.shape[0]

        for i in range(nrows):
            cog_arr[i] = np.sum(arr_output[:,i] * lin2) / np.sum(arr_output[:,i])
 
        cf_p, sigmaa = curve_fit(sine_wave, x, cog_arr, p0=cf_p)
        cog_fit = sine_wave(x, cf_p[0], cf_p[1], cf_p[2])

        cog_mean = a[0] - 1.
        error_after = cog_mean - (ncols-1)/2.
        if debug:
            print(f'Fitted center of gravity after correction= {cog_mean:8.2f}, +-{sigmaa[0]:8.2f}')
            print(f'Error after correction (offset from center of image) = {error_after:8.2}f')


    cog_arr = np.array([cog_arr, cog_fit]).T
    if return_cog:
        return cog_arr

    if debug:
        print('Sinogram used average of '+str(air_values)+" pixels for air")
        print('Skipped '+ str(acc_values)+ ' acceleration pixels')

    """
    if backlash and debug:
        print('Backlash corrected '+str(back)+' pixels')
    """
        
    if auto_center and debug:
        print('Center corrected '+str(shift_amount)+ ' pixels Absolute center = '\
              + str(center) +  ' pixels')

    return arr_output




def correct_rotation_axis(vol, max_error=10):
    """
    This procedure corrects for rotation axis wobble. It works as follows: 
     - Computes the sinogram and center of gravity for each slice
     - Computes the errors (fitted COG minus actual COG for each angle
     - Averages the errors as a function of angle
     - Shifts each angle to correct for the average error
     - The calculation is only done for slices in which the maximum error in the fit is less
       than max_error pixels (default=10).
    
    Parameters
    ----------
    infile : TYPE
        DESCRIPTION.
    outfile : TYPE
        DESCRIPTION.
    max_error : TYPE, optional
        DESCRIPTION. The default is .

    Returns
    -------
    None.

    """

    nx, ny, nangles = vol.shape


    ngood = 0
    errs = np.zeros([nangles, ny])
    angles = np.linspace(0, 360, nangles, endpoint=False)

    print('\nDetermining rotation errors')
    for i in range(ny):
        tomo_slice = vol[:,i,:]

        cog = sinogram(tomo_slice, angles, return_cog=True, debug=True, fluorescence=True)
        return cog
        y = cog[:,0]
        yfit = cog[:,1]

        err = yfit - y
        #Don't count slices for which:
        #- the maximum error in the fitted center of gravity is more than max_error pixels
        if np.max(np.abs(err)) >= max_error:
            errs[:,i] = 0.0
        else:
            errs[:,i] = yfit-y
            ngood += 1

        print('Slice=', i, 'max(err) =', np.max(np.abs(err)), \
          ' max(errs[*,i])=', np.max(errs[:,i]), ' min(errs[*,i])=', np.min(errs[:,i]))

    """
    #image_display, errs, min=-10, max=10
    t = np.sum(errs, axis=1) / ngood
    sh = np.round(t)
    v = vol
    for i in range(nangles):
        print('shifting angle ', i, ' by ', sh[i])
        v[:,:,i] = np.roll(vol[:,:,i], sh[i], 0)
    """
    return cog