import numpy as np

def MakeRadiusImage(nPixels=(512,512), dPixel=(1,1), center=(0,0)):

    #Error Control
    if(np.size(nPixels) > 2):
       print("nPixels may only have 1 or 2 variables")
    elif(np.size(nPixels) == 1):
        nPixels = np.repeat(nPixels, 2)
    else:
        nPixels = np.array(nPixels)
              
    if(np.size(dPixel) > 2):
       print("dPixel may only have 1 or 2 variables")
    elif(np.size(dPixel) == 1):
        dPixel = np.repeat(dPixel, 2)
    else:
        dPixel = np.array(dPixel)

    if(np.size(center) > 2):
       print("center may only have 1 or 2 variables")
    elif(np.size(center) == 1):
        center = np.repeat(center, 2)
    else:
        center = np.array(center)

    #Compute edge
    edge = dPixel*(nPixels-1)/2.0 - center
    
    x = np.linspace(-1.0*edge[0],edge[0],nPixels[0])
    y = np.linspace(-1.0*edge[0],edge[0],nPixels[1])
                
    xv,yv = np.meshgrid(x,y)
    
    return np.sqrt(xv**2 + yv**2)


def MakePhiImage(nPixels=(512,512), dPixel=(1,1), center=(0,0)):

    #Error Control
    if(np.size(nPixels) > 2):
       print("nPixels may only have 1 or 2 variables")
    elif(np.size(nPixels) == 1):
        nPixels = np.repeat(nPixels, 2)
    else:
        nPixels = np.array(nPixels)
              
    if(np.size(dPixel) > 2):
       print("dPixel may only have 1 or 2 variables")
    elif(np.size(dPixel) == 1):
        dPixel = np.repeat(dPixel, 2)
    else:
        dPixel = np.array(dPixel)

    if(np.size(center) > 2):
       print("Center may only have 1 or 2 variables")
    elif(np.size(center) == 1):
        center = np.repeat(center, 2)
    else:
        center = np.array(center)

    #Compute edge
    edge = dPixel*(nPixels-1)/2.0 - center
    
    x = np.linspace(-1.0*edge[0],edge[0],nPixels[0])
    y = np.linspace(-1.0*edge[1],edge[1],nPixels[1])
                
    xv,yv = np.meshgrid(x,y)
    
    return np.arctan2(yv, xv)

