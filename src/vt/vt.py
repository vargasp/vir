import numpy as np
import matplotlib as mpl

from PIL import Image
Image.MAX_IMAGE_PIXELS = None


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

from mpl_toolkits.axes_grid1 import make_axes_locatable

#Sets the default font for all images
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.family'] ='Roboto'
plt.rcParams['font.family'] ='Times New Roman'


def round_sig(x, sig):
    """
    Rounds a number x to sig number of significant digits 

    Parameters
    ----------
    x : int or float
        Value to be rounded.
    sig : int
        The number of significant digits to round x.

    Returns
    -------
    float
        Rounded value.
    """
    
    #Condition to account for the log of 0
    if x == 0.0:
        return 0.0
    else:
        return round(x, sig-int(np.floor(np.log10(abs(x))))-1)


def float2readablestr(x, sig=3):
    """
    Converts a float or int to a readable string value

    Parameters
    ----------
    x : int or float
        Value to be converted.
    sig : int
        The number of significant digits to round x. Default is 3

    Returns
    -------
    str
        A human readable string of x.
    """

    #Rounds the variable to sig number of signficant digits
    x = round_sig(x, sig)
    
    #If the value is within machine precission to 0 return '0'
    if np.isclose(x, 0):
        return "{:d}".format(int(np.rint(x)))

    return "{:g}".format(x)


#Manipulates the y-axis and x-axis to be the correct dimensions
def FormatData(xs, ys):

    #Format y-axis
    ys = np.array(ys)
    if(ys.ndim > 2 or ys.ndim == 0):
       raise ValueError("Y array must have a dimension of 1 or 2")
    elif(ys.ndim == 1):
        ys = ys[:,np.newaxis]

    #Format x-axis
    nSamples, nPlots = ys.shape
    if xs is None:
        xs = np.arange(nSamples)
    else:
        xs = np.array(xs)

    if(xs.ndim > 2 or xs.ndim == 0):
        raise ValueError("X array must have a dimension of 1 or 2")
    elif(xs.ndim == 1):
        xs = np.tile(xs, [nPlots,1]).T

    return xs, ys


def FormatLabels(labels, nLabels):

    if labels is None:
        labels = np.repeat('',nLabels)
    
    if nLabels == 1:
        labels = [labels]

    return labels

def FormatLines(color, marker, linestyle, nPlots):
    if color is None:
        color = [None]*nPlots
        
    if len(marker) != nPlots:
        marker = [marker]*nPlots

    if len(linestyle) != nPlots:
        linestyle = [linestyle]*nPlots

    return color, marker, linestyle


def FormatGrid(ax, glidlines, axis):
        if axis == "x":
            ax.grid(which=glidlines, axis=axis, linestyle='-', linewidth='0.25',color='k')
        elif axis == "y":
            ax.grid(which=glidlines, axis=axis, linestyle='-', linewidth='0.25',color='k')
        else:
            ax.grid(which=glidlines, linestyle='-', linewidth='0.25',color='k')


def FormatAxes(ax,xlim,ylim,scale):
    ax.set_xscale(scale[0])
    ax.set_yscale(scale[1])
    
    if xlim != None:
        ax.set_xlim(xlim[0],xlim[1])
    
    if ylim != None:
        ax.set_ylim(ylim[0],ylim[1])


def FormatPlotTicks(ax,xticks,yticks):
    
    if xticks is not None:
        ax.set_xticks(xticks)
      
    if yticks is not None:
        ax.set_yticks(yticks)
      
    
def FormatTicks(ax,image,xlim,ylim,ticks):
    
    if xlim is not None:
        ticks = True
        ticks_loc = ax.get_xticks().tolist()
        tick_labels = np.array(ticks_loc)/(image.shape[1])* \
            (xlim[1] - xlim[0]) + xlim[0]
        
        """
        print(np.array(ticks_loc))
            
            
        print("HI")
        print(ax.get_xlim())
        k = ax.get_xlim()
        ax.set_xlim(xlim)
        ticks_loc = ax.get_xticks().tolist()
        ax.set_xlim(k)
        
        print(ax.get_xlim())
        
        print(np.array(ticks_loc))
        print(image.shape[1])
        print(xlim[1] - xlim[0])
        print(tick_labels)
        """
        
        tick_labels = [float2readablestr(x) for x in tick_labels]
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_labels))
        
    if ylim is not None:
        ticks = True
        ticks_loc = ax.get_yticks().tolist()
        tick_labels = np.array(ticks_loc)/(image.shape[0]-1)* \
            (ylim[1] - ylim[0]) + ylim[0]

        tick_labels = [float2readablestr(x) for x in tick_labels]
        ax.yaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(tick_labels))

    if ticks == False:
        RemoveTicks(ax)

def RemoveTicks(ax):
    ax.tick_params(length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')


def FormatColorBar(ax,im,vmin,vmax,ctitle):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks = [vmin,vmax])
    cbar.ax.set_ylabel(ctitle)
    cbar.ax.tick_params(labelsize=10) 


def DisplayPlot(fig, outfile):
    if outfile:
        plt.savefig(outfile + '.png',format='png',bbox_inches='tight',dpi=1000)
        plt.close(fig)
    else:
        plt.show()


def LabelPlot(ax,title,xtitle,ytitle):
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(10)
        tick.set_color('black')

    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(10)
        tick.set_color('black')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xtitle, fontsize=12)
    ax.set_ylabel(ytitle, fontsize=12)
    
  
def scale_bar(ax, image, pixels, sb_label=u"10 \u03bcm"):
    nY, nX = image.shape
    x0 = (0.95 - pixels/nX) *nX
    x1 = 0.95*nX
    y0 = 0.05*nY
    y1 = 0.0625*nY
    text_shfit = 0
    
    plt.plot([x0,x1], [y0,y0], 'w-')
    plt.plot([x0,x0], [y0,y1], 'w-')
    plt.plot([x1,x1], [y0,y1], 'w-')
    
    ax.text(x0+text_shfit, 0.01*nY,sb_label, fontsize=10, color='white')


def CreatePlot(ys,xs=None,err=None,title="",xtitle="",ytitle="",\
    xlims=None,ylims=None,xticks=None,yticks=None,\
    scale=("linear","linear"),grid=False,grid_minor=False,\
    color=None,marker='',linestyle='-',labels=None,outfile=False):
    """
    Displays/Creates a plot. This is a wrapper function for matplotlib.pyplot

    Parameters
    ----------
    ys : (samples, nPlots) or (samples) array_like
        The y values that are plotted.
    xs : (samples, nPlots) or (samples) array_like, optional
        The x values that are plotted.. The default is None.
    err : TYPE, optional
        DESCRIPTION. The default is None.
    title : string, optional
        The title of the plot. The default is "".
    xtitle : string, optional
        The x-axis title of the plot. The default is "".
    ytitle : string, optional
        The y-axis title of the plot. The default is "".
    xlims : (2) array_like, optional
        The range. The default is None.
    ylims : TYPE, optional
        DESCRIPTION. The default is None.
    scale : TYPE, optional
        DESCRIPTION. The default is ("linear","linear").
    grid : TYPE, optional
        DESCRIPTION. The default is False.
    grid_minor : TYPE, optional
        DESCRIPTION. The default is False.
    color : TYPE, optional
        DESCRIPTION. The default is None.
    marker : TYPE, optional
        DESCRIPTION. The default is ''.
    linestyle : TYPE, optional
        DESCRIPTION. The default is '-'.
    labels : TYPE, optional
        DESCRIPTION. The default is None.
    outfile : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """

    xs, ys = FormatData(xs, ys)
    
    nSamples, nPlots = ys.shape
    labels = FormatLabels(labels, nPlots)
    color, marker, linestyle = FormatLines(color, marker, linestyle, nPlots)

    fig, ax = plt.subplots()
    for i in range(nPlots):
        if err is None:
            ax.plot(xs[:,i],ys[:,i],label=labels[i],marker=marker[i],\
                    linestyle=linestyle[i],color=color[i])
        else:
            ax.errorbar(xs[:,i],ys[:,i],yerr=err,label=labels[i],\
                        fmt='o',capsize=3,markersize=3,\
                        capthick=.5,elinewidth=.5,ecolor='r')

    if labels[0] != '': ax.legend(loc=0, fontsize=10)
    if grid != False: FormatGrid(ax,'major', grid)
    if grid_minor != False: FormatGrid(ax,'minor', grid)

    FormatAxes(ax,xlims,ylims,scale)
    FormatPlotTicks(ax,xticks,yticks)
    LabelPlot(ax,title,xtitle,ytitle)
    DisplayPlot(fig,outfile)


def array2list(arr):
    l = []
    for i in range(arr.shape[1]):
        l.append(arr[:np.count_nonzero(arr[:,i]),i])

    return l


def CreateBoxPlot(samples,title="",xtitle="",ytitle="",labels="",outfile=False,remove_zeros=False,showfliers=True):
    '''
    samples (samples, expriments)
    
    '''

    #ys = FormatYs(ys)

    if remove_zeros == True:
        samples = array2list(samples)

    fig, ax = plt.subplots()
    ax.boxplot(samples,showfliers=showfliers)

    LabelPlot(ax,title,xtitle,ytitle)
    ax.set_xticklabels(labels)
    DisplayPlot(fig,outfile)


def CreateTiffImage(image, outfile=None):
    
    if image.ndim == 2:
        img = Image.fromarray(image.astype(np.float32))
        img.save(outfile+'.tif')


    if image.ndim == 3:
        
        n = image.shape[2]
        pad = int(np.log10(abs(n))) + 1
        
        for i in range(n):
            img = Image.fromarray(image[:,:,i].astype(np.float32))
            img.save(outfile + str(i).zfill(pad) +'.tif')



def save_png(img,outfile,vmax=None,vmin=None):
    if vmin is None:
        vmin = img.min()
    if vmax is None:
        vmax = img.max()
        
    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")
        
        
    # Make one working copy (avoid modifying the original array)
    arr = np.array(img, dtype=np.float32, copy=True)

    # Normalize in-place
    np.clip(arr, vmin, vmax, out=arr)
    np.subtract(arr, vmin, out=arr)
    np.multiply(arr, 65535.0 / (vmax - vmin), out=arr)

    # Convert to 16-bit
    arr = arr.astype(np.uint16)

    # Save with faster PNG compression
    Image.fromarray(arr).save(outfile)


def ReadTiffImageStack(infile, z):
    im_obj = Image.open(infile)
    im_obj.seek(z)

    return np.array(im_obj)
        

def ReadTiffImage(infile):
    im_obj = Image.open(infile)
    n_frames = im_obj.n_frames
    
    if n_frames == 1:
        return np.array(im_obj)
    else:
        w,h = im_obj.size
        dataset = np.empty([h,w,n_frames])
        
        for i in range(n_frames):
           im_obj.seek(i)
           dataset[:,:,i] = np.array(im_obj)
        
        return dataset


def CreateImage(image,window=False,title ="",xtitle="",ytitle="",ctitle="",\
                xvals=None,yvals=None,coords=None,outfile=None,text_strings=None,
                aspect='equal',cmap="gray"):
    """
    Displays/Creates a BW intensity image

    Parameters:        
        image:    2d np.array of the image to be displayed
        window:   list or tuple of window range (min, max)
        title:    title for the image
        ctitle:   title for the colorbar
        outfile:  location to save the file
                
    Returns:
        Nothing
    """
    plt.rcParams['figure.dpi'] = 600
    
    if window == False:
        window = [np.min(image),np.max(image)]

    fig, ax = plt.subplots()
    #im = plt.imshow(image, vmin=window[0], vmax=window[1])
    im = plt.imshow(image, vmin=window[0], interpolation='None', vmax=window[1], cmap=cmap, extent=coords, origin="lower")
    LabelPlot(ax,title,xtitle,ytitle)
    
    if coords == None:
        RemoveTicks(ax)
    

    if ctitle != "":
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks=window)
        cbar.ax.set_ylabel(ctitle)
        cbar.ax.tick_params(labelsize=10) 

    ax.set_aspect(aspect)
    if text_strings != None:
        for text_string in text_strings:
            ax.text(text_string[0], text_string[1], text_string[2], fontsize=10, color='white')



    DisplayPlot(fig,outfile)



def CreateMotif(images,ncol,nrow,window=False,titles ="",xtitle="",ytitle="",ctitle="",\
                xvals=None,yvals=None,coords=None,outfile=None,text_strings=None,
                aspect='equal',cmap="gray"):
    """
    Displays/Creates a BW intensity image

    Parameters:        
        image:    2d np.array of the image to be displayed
        window:   list or tuple of window range (min, max)
        title:    title for the image
        ctitle:   title for the colorbar
        outfile:  location to save the file
                
    Returns:
        Nothing
    """
    plt.rcParams['figure.dpi'] = 600
    
    if window == False:
        window = [np.min(images),np.max(images)]


    fig, axes = plt.subplots(nrow,ncol)


    # Flatten the axes array for easy iteration


    # Iterate through images and plot them
    for i, (ax, img) in enumerate(zip(axes.flatten(), images)):
        ax.imshow(img, vmin=window[0], interpolation='None', vmax=window[1], cmap=cmap, extent=coords, origin="lower")
        
        LabelPlot(ax,titles[i],xtitle,ytitle)
    
        if coords == None:
            RemoveTicks(ax)
    
        ax.set_aspect(aspect)
        if text_strings != None:
            for text_string in text_strings:
                ax.text(text_string[0], text_string[1], text_string[2], fontsize=10, color='white')

    DisplayPlot(fig,outfile)



def imshow(image,vmin=None,vmax=None,title ="",xtitle="",ytitle="",ctitle="",\
           cmap=cm.Greys_r,ticks=False,xlim=None,ylim=None,extent=None,outfile=None):
    """
    Displays/Creates a BW intensity image

    Parameters:        
        image:    2d np.array of the image to be displayed
        window:   list or tuple of window range (min, max)
        title:    title for the image
        ctitle:   title for the colorbar
        outfile:  location to save the file
                
    Returns:
        Nothing
    """

    cmap = 'inferno'
    if vmin is None:
        vmin = np.min(image)

    if vmax is None:
        vmax = np.max(image)    

    fig, ax = plt.subplots()
    im = plt.imshow(image, vmin=vmin, vmax=vmax, interpolation='None', \
                    cmap=cmap, extent=extent, origin="lower")
    LabelPlot(ax,title,xtitle,ytitle)
    FormatTicks(ax,image,xlim,ylim,ticks)

    if ctitle != "":
        FormatColorBar(ax,im,vmin,vmax,ctitle)

    DisplayPlot(fig,outfile)


def imshow2(image,vmin=None,vmax=None,title ="",xtitle="",ytitle="",ctitle="",\
           cmap=cm.Greys_r,ticks=False,xlim=None,ylim=None,extent=None,outfile=None):
    """
    Displays/Creates a BW intensity image

    Parameters:        
        image:    2d np.array of the image to be displayed
        window:   list or tuple of window range (min, max)
        title:    title for the image
        ctitle:   title for the colorbar
        outfile:  location to save the file
                
    Returns:
        Nothing
    """

    cmap = 'inferno'
    if vmin is None:
        vmin = np.min(image)

    if vmax is None:
        vmax = np.max(image)    

    fig, ax = plt.subplots()
    im = plt.imshow(image, vmin=vmin, vmax=vmax, interpolation='None', \
                    cmap=cmap, extent=extent, origin="lower")
    #FormatTicks(ax,image,xlim,ylim,ticks)

    
    # Minor ticks
    #ax.set_xticks(np.arange(-.5, image.shape[1], 1), minor=True)
    #ax.set_yticks(np.arange(-.5, image.shape[0], 1), minor=True)
    
    # Gridlines based on minor ticks
    #ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))

    ax.xaxis.set_minor_locator(ticker.IndexLocator(base=1.0,offset=0.0))
    ax.yaxis.set_minor_locator(ticker.IndexLocator(base=1.0,offset=0.0))
    ax.xaxis.set_major_locator(ticker.IndexLocator(base=1.0,offset=0.5))
    ax.yaxis.set_major_locator(ticker.IndexLocator(base=1.0,offset=0.5))

    ax.grid(which='minor', color='k', linestyle='-', linewidth=.5)
    #ax.grid(which='major', color='k', linestyle='-', linewidth=.5)

    #RemoveTicks(ax)
    DisplayPlot(fig,outfile)


def hist_show(arr, window=False, bins=None, title="", xtitle="", ytitle="",ctitle="",\
              xlabels=None, ylabels=None,outfile=False):

    if bins == None:
        binsX, binsY = arr.shape
    else:
        if arr.shape[0]%bins[0] != 0 or arr.shape[1]%bins[1] != 0:
            raise ValueError("Array must be evenly divisible by bins")
        else:
            binsX, binsY = bins

    if window == False:
        window = [np.min(arr),np.max(arr)]
            
    fig, ax = plt.subplots()
    im = plt.imshow(arr.T, origin='lower',cmap=cm.Greys_r)
    LabelPlot(ax,title,xtitle,ytitle)
    
    bin_samplesX = int(arr.shape[0]/binsX)
    bin_samplesY = int(arr.shape[1]/binsY)

    #Sets major X ticks/ticklabels to be the grid
    ax.set_xticks(np.linspace(-.5, binsX*bin_samplesX-.5, binsX+1))
    ax.set_xticklabels('')
    
    if xlabels is not None:
        ax.set_xticks(np.linspace(bin_samplesX/2.0, bin_samplesX*(binsX - 0.5), binsX) - 0.5, minor=True)    
        ax.set_xticklabels(xlabels, minor=True)
    
    #Sets major X ticks/ticklabels to be the grid
    ax.set_yticks(np.linspace(-.5, binsY*bin_samplesY-.5, binsY+1))
    ax.set_yticklabels('')

    if ylabels is not None:
        ax.set_yticks(np.linspace(bin_samplesY/2.0, bin_samplesY*(binsY - 0.5), binsY) - 0.5, minor=True)
        ax.set_yticklabels(ylabels,minor=True)
   
    ax.tick_params(axis='both', which='minor', length=0.0,labelsize=6)    
    ax.grid(visible=True, which='Major', linestyle='-', linewidth='0.5',color='k')
    
    
    if ctitle != "":
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = plt.colorbar(im, cax=cax, ticks=window)
        cbar.ax.set_ylabel(ctitle)
        cbar.ax.tick_params(labelsize=10) 

    DisplayPlot(fig,outfile)


def animated_gif(images, outfile, fps=24.):
    """
    Creates an animated gif from a stack of images

    Parameters
    ----------
    images : (n, nY, nX)
        Stack of images to animate.
    outfile : string
        The outputfile name
    fps : float, optional
        The number of images displayed per second. The default is 24.

    Returns
    -------
    None.

    """
    
    gif = []
    for image in images:
        image = image.clip(0)*255/images.max().T
        img = Image.fromarray(image.astype(np.float32))
        gif.append(img)

    print(len(gif)/fps)
    gif[0].save(outfile+'.gif', save_all=True,optimize=False, \
                append_images=gif[1:], duration=len(gif)/fps, loop=0)




