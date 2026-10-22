import numpy as np
import projection as pj
import filtration as ft
import time
import sys

def fan_fbp(Sinogram, src_iso, src_det, equispaced=False, \
            filter_name='ramp', cutoff=0.5, \
            nPixels=(512,512), dPixels=(1.0,1.0), pixOff = (0.0,0.0), \
            dDet=1.0, detOff=0.0, \
            coverage=2.0*np.pi, proj0=0.0, rotation='CCW', projOff=0.0, \
            Hsieh = True):
    '''
    Filtered back-projection algorithm for fan-beam data, based on Kak and Slaney
    
    Input:
            Sinogram: 2d numpy array with shape [nViews,nBins]
            src_iso: Distance from source to isocenter
            src_det: Distance from source to detector
            detOff: Offset of the center of rotation from the midline of the detector array []
            start_angle: Angle of first projection in degrees            
            coverage: Total angular coverage of views in degrees
            roation: Rotation direction in superior-inferior view 
            N: Output size of image (NxN)
            equalspace is not impleletned
    '''

    if len(Sinogram.shape) == 2:
        Sinogram = Sinogram[:,None,:]
        
    nProjs,nRows,nCols = Sinogram.shape

    if equispaced:
        dCol = dDet/src_det*src_iso #Effective distance of detector at isocenter (units)
    else:
        dCol = float(dDet)/src_det #Fan angle per detector in (radians)

    #Create a vector of source angular positions
    Views, Rows, Cols = calc_geom_arrays(Sinogram.shape, \
         coverage=coverage, rotation=rotation, proj0=proj0, projOff=projOff, \
         dCol=dCol)
    
    #Cosine weight sinogram
    #start = timeprint('Weighting sinogram')
    Sinogram = cosine_weighting(Sinogram, Cols, src_iso, detOff=detOff, equispaced=equispaced)
    #timeprint('Weighting sinogram',start=start)
    
    #Filter Sinogram
    #start = timeprint('Filtering sinogram')
    Sinogram = ft.filter_sino(Sinogram, dCol=dCol, filter_name=filter_name)    
    #timeprint('Filtering sinogram',start=start)
    
    #Backproject the filtered sinogram
    #start = timeprint('Backprojecting sinogram')
    image = pj.fan_backproject_hsieh(Sinogram, Views, Cols, src_iso, \
        dPixels=dPixels, nPixels=nPixels, detOff=detOff, pixOff=pixOff)
    #timeprint('Backprojecting sinogram',start=start)
	
    #Now normalize by factor delta_beta=2.0*!PI/nanglesand rotate image because
    #the definition of beta used in generating projections is offset by !PI/2.
    
    #image = 2.0*np.pi*image/float(nAngles)
    image = np.pi*image/(2*Views.size)

    image = image*mask_circle(img_shape=nPixels)

    return image


def calc_geom_arrays(sino_shape, nViews=None, coverage=2.0*np.pi, \
        rotation='CCW', dProj=None, proj0=0.0, projOff=0.0, \
        dCol=1.0, dRow=1.0):

    nProjs,nRows,nCols = sino_shape
    
    if nViews == None:
        nViews = nProjs
    
    if dProj == None:        
        dProj = coverage/nViews
    
    if rotation == 'CCW':
        rotation = 1.0
    elif rotation == 'CW':
        rotation = -1.0
    else:
        print('Rotation Error')
        
    #Calculate detector array element coordinates
    Cols = np.linspace(1-nCols,nCols-1,nCols)*dCol/2.0
    Rows = np.linspace(1-nRows,nRows-1,nRows)*dRow/2.0
    
    #Calculate projection array coordinates
    Projs = rotation*(np.arange(nProjs)*dProj + projOff) + proj0 

    return Projs, Rows, Cols
    

def cosine_weighting(Sino, Cols, src_iso, detOff=0.0, equispaced=False):

    #Create vectors of souce bin locations and corresponding weights
    if(equispaced):
        Weights = (src_iso - detOff*Cols/src_iso) / np.sqrt(src_iso**2 + Cols**2)
    else:
        Weights = src_iso*np.cos(Cols) - detOff*np.sin(Cols)

    return Weights*Sino
    

def mask_circle(img_shape=(512,512), center=None, radius=None):
    """
    Returns a boolean array with a cicular mask with the specifiec paramters
        
    Parameters:
    
        center:    The location of the center of the mask (j,i)  [index]
        radius:    The size of the mask's radius in pixels
        img_shape: List or tuple specifying pixels [nrows,ncols] for the image
       
        Returns:
        
        A boolean array with a cicular mask
    """

    if center == None:
        center = ()
        center += (img_shape[0]/2,)
        center += (img_shape[1]/2,)
    
    if radius == None:
        radius = np.min(img_shape)/2.0*.95
    
    x,y = np.indices((img_shape[0],img_shape[1]))
    
    return (x - center[0])**2 + (y - center[1])**2 < radius**2


"""
def timeprint(string,start=None):
    if start == None:
        sys.stdout.write(string)
        sys.stdout.flush()
        return time.clock()
    else:
        sys.stdout.write('\r' + string + ' -- ' +str(time.clock()-start)+ ' secs\n')
        sys.stdout.flush()
"""        

