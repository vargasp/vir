/*
NAME:
proj.c

PURPOSE:
A library of projection functions

CATEGORY:
Ray Tracing

MODIFICATION HISTORY:
Phillip Vargas, December 2021
 */

#include <math.h>
#include <stdio.h>

/*
 NAME:
 back_proj_ray
 
 PURPOSE:
 Backprojects a single vector onto an array.
 
 CALLING SEQUENCE:
 back_proj_ray(*phantom, *sino, *pix, *length, nElem, sinoElemIdx);
 
 INPUTS:
 float *phantom:    A flattened grid where the vector will be projected
 float *sino:       A flattened array of bins that contain the line integral values of the vector
 int *pix:          An array of voxel indices of the vector that map to the phantom
 float *lengt       An array of intersection lengths of the vector
 int nElem:         The number of elements in the vector
 int sinoElemIdx:   The sinogram index assocatite with the vector
 */
 
/* 
         printf("n: %d, i: %d, ray->L[i]: %f\n", ray->n, i, ray->L[i]);
         printf("x: %d, y: %d, z: %d, p_idx: %d\n", ray->X[i], ray->Y[i], ray->Z[i], phantIdx);
         printf("Value: %f, Idx: %d, Sino[Idx]: %f\n\n", phantom[phantIdx]*ray->L[i],sinoElemIdx, sino[sinoElemIdx]);
*/



// Compilation and linking:
// gcc -c -Wall -Werror -fpic projection_c.c
// gcc -shared -o projection_c.so projection_c.o



typedef struct Ray{
    int n;
    int *X;
    int *Y;
    int *Z;
    float *L;
} Ray;


typedef struct Coord{
    int x;
    int y;
    int z;
} Coord;


int imin(int x, int y)
{
    return (x > y) ? y : x;
}


int imax(int x, int y)
{
    return (x < y) ? y : x;
}


/*Ravels a 3d coordinates indices to a flat 1d index*/
int ravel(int x, int y, int z, int nX, int nY, int nZ)
{
    return nY*nZ*x + nZ*y + z;
}

/*Ravels a 3d coordinates indices to a flat 1d index with structure parameters*/
int ravel_s(int x, int y, int z, Coord *dims)
{
    return dims->y*dims->z*x + dims->z*y + z;
}


void forward_proj_ray(float *phantom, float *sino, int *pix, float *length, int nElem, int sinoElemIdx)
{
    int i;
    
    for(i=0; i<nElem; i++){
        sino[sinoElemIdx] += phantom[pix[i]]*length[i];
    }
}


void back_proj_ray(float *phantom, float *sino, int *pix, float *length, int nElem, int sinoElemIdx)
{
    int i;
    
    for(i=0; i<nElem; i++){
      phantom[pix[i]] += length[i]*sino[sinoElemIdx];
    }
}

void forward_proj_ray_u(float *phantom, float *sino, int *X, int *Y, int *Z, float *length, int nElem, int nX, int nY, int nZ, int sinoElemIdx)
{
    int i;
    int phantIdx;
    
    for(i=0; i<nElem; i++){
        phantIdx = ravel(X[i], Y[i], Z[i], nX, nY, nZ);
        sino[sinoElemIdx] += phantom[phantIdx]*length[i];
    }
}


void back_proj_ray_u(float *phantom, float *sino, int *X, int *Y, int *Z, float *length, int nElem, int nX, int nY, int nZ, int sinoElemIdx)
{
    int i;
    int phantIdx;
    
    for(i=0; i<nElem; i++){
        phantIdx = ravel(X[i], Y[i], Z[i], nX, nY, nZ);
        phantom[phantIdx] += length[i]*sino[sinoElemIdx];
    }
}


void forward_proj_ray_u_struct(float *phantom, float *sino, Ray *ray, Coord *dims, int sinoElemIdx)
{
    int i;
    int phantIdx;
        
    for(i=0; i<ray->n; i++){
        phantIdx = ravel_s(ray->X[i], ray->Y[i], ray->Z[i], dims);
        sino[sinoElemIdx] += phantom[phantIdx]*ray->L[i];
    }
}


void back_proj_ray_u_struct(float *phantom, float *sino, Ray *ray, Coord *dims, int sinoElemIdx)
{
    int i;
    int phantIdx;
    
    for(i=0; i<ray->n; i++){
        phantIdx = ravel_s(ray->X[i], ray->Y[i], ray->Z[i], dims);
        phantom[phantIdx] += sino[sinoElemIdx]*ray->L[i];
    }
}


void forward_proj_rays_u_struct(float *phantom, float *sino, Ray *ray, Coord *dims, int nRays)
{
    int j;
    for(j=0; j<nRays; j++){
        forward_proj_ray_u_struct(phantom, sino, ray, dims, j);
        ray++;
    }
}


void back_proj_rays_u_struct(float *phantom, float *sino, Ray *ray, Coord *dims, int nRays)
{
    int j;

    for(j=0; j<nRays; j++){
        back_proj_ray_u_struct(phantom, sino, ray, dims, j);
        ray++;
    }
}


void forward_proj_ray_t_u(float *phantom, float *sino, Ray *ray, Coord *dims, Coord *bins0, Coord *binsN)
{
    int j,k,l;
    int phantIdx, sinoIdx;
    int x_idx, y_idx;
    int nBinsZ;
    
    nBinsZ = binsN->z - bins0->z;

    for(l=0; l<ray->n; l++){
        x_idx = dims->y*dims->z*ray->X[l];
        
        for(j=imax(0,ray->Y[l] + bins0->y); j<imin(ray->Y[l] + binsN->y, dims->y); j++){
            y_idx = dims->z*j;

            for(k=imax(0,ray->Z[l] + bins0->z); k<imin(ray->Z[l] + binsN->z,dims->z); k++){
                phantIdx = x_idx + y_idx + k;

                sinoIdx = nBinsZ*(j - bins0->y - ray->Y[l]) + (k - bins0->z - ray->Z[l]);                        
                sino[sinoIdx] = phantom[phantIdx]*ray->L[l];
            }
        }
    }
}


void back_proj_ray_t_u(float *phantom, float *sino, Ray *ray, Coord *dims, Coord *bins0, Coord *binsN)
{
    int j,k,l;
    int phantIdx, sinoIdx;
    int x_idx, y_idx;
    int nBinsZ;
    
    nBinsZ = binsN->z - bins0->z;

    for(l=0; l<ray->n; l++){
        x_idx = dims->y*dims->z*ray->X[l];
        
        for(j=imax(0,ray->Y[l] + bins0->y); j<imin(ray->Y[l] + binsN->y, dims->y); j++){
            y_idx = dims->z*j;

            for(k=imax(0,ray->Z[l] + bins0->z); k<imin(ray->Z[l] + binsN->z,dims->z); k++){
                phantIdx = x_idx + y_idx + k;

                sinoIdx = nBinsZ*(j - bins0->y - ray->Y[l]) + (k - bins0->z - ray->Z[l]);                        
                phantom[phantIdx] += sino[sinoIdx]*ray->L[l];
            }
        }
    }
}


void forward_proj_rays_t_u(float *phantom, float *sino, Ray *ray, Coord *dims, Coord *bins0, Coord *binsN, int nRays)
{
    
    int j,k,l,m;
    int p_idx_x, p_idx_xy;
    int s_idx0, s_idx_m, s_idx_l, s_idx_lj;
    int nBinsY, nBinsZ;
    int k0,kN;

    nBinsY = binsN->y - bins0->y;
    nBinsZ = binsN->z - bins0->z;
    
    s_idx0 = nBinsZ*bins0->y + bins0->z;
    
    for(m=0; m<nRays; m++){
        s_idx_m = m*nBinsY*nBinsZ - s_idx0;
        
        for(l=0; l<ray->n; l++){
            p_idx_x = dims->y*dims->z*ray->X[l];
            s_idx_l = s_idx_m - nBinsZ*ray->Y[l] - ray->Z[l];
            
            k0 = imax(0,ray->Z[l] + bins0->z);
            kN = imin(ray->Z[l] + binsN->z, dims->z);
            
            for(j=imax(0,ray->Y[l] + bins0->y); j<imin(ray->Y[l] + binsN->y, dims->y); j++){
                p_idx_xy = p_idx_x + dims->z*j;
                s_idx_lj = s_idx_l + nBinsZ*j;
    
                for(k=k0; k<kN; k++){
                    sino[s_idx_lj + k] = phantom[p_idx_xy + k]*ray->L[l];
                }
            }
        }
        
        ray++;
    }
}



void back_proj_rays_t_u(float *phantom, float *sino, Ray *ray, Coord *dims, Coord *bins0, Coord *binsN, int nRays)
{
    
    int j,k,l,m;
    int phantIdx, sinoIdx;
    int x_idx, xy_idx, s_idx;
    int nBinsY, nBinsZ;
    
    int k0,kN;
    
    
    nBinsY = binsN->y - bins0->y;
    nBinsZ = binsN->z - bins0->z;

    
    for(m=0; m<nRays; m++){
        s_idx = m*nBinsY*nBinsZ;

        for(l=0; l<ray->n; l++){
            x_idx = dims->y*dims->z*ray->X[l];
            
            k0 = imax(0,ray->Z[l] + bins0->z);
            kN = imin(ray->Z[l] + binsN->z, dims->z);
            
            for(j=imax(0,ray->Y[l] + bins0->y); j<imin(ray->Y[l] + binsN->y, dims->y); j++){
                xy_idx = dims->z*j + x_idx;
                
                for(k=k0; k<kN; k++){
                    
                    sinoIdx = s_idx + nBinsZ*(j - bins0->y - ray->Y[l]) + (k - bins0->z - ray->Z[l]);
                    phantIdx = xy_idx + k;
                    phantom[phantIdx] += sino[sinoIdx]*ray->L[l];
                }
            }
            /*
            for(j=imax(0,ray->Y[l] + bins0->y); j<imin(ray->Y[l] + binsN->y, dims->y); j++){
                xy_idx = dims->z*j + x_idx;
    
                for(k=imax(0,ray->Z[l] + bins0->z); k<imin(ray->Z[l] + binsN->z,dims->z); k++){
    
                    sinoIdx = s_idx + nBinsZ*(j - bins0->y - ray->Y[l]) + (k - bins0->z - ray->Z[l]);
                    phantIdx = xy_idx + k;
                    phantom[phantIdx] += sino[sinoIdx]*ray->L[l];
                }
            }
            
            */
            
        }
        
        ray++;
    }
}



void forward_proj_rays_t_sym_u(float *phantom, float *sino, Ray *ray, Coord *dims, Coord *bins0, Coord *binsN, int nRays)
{
    
    int j,k,l,m;
    int phantIdx, sinoIdx;
    int x_idx, y_idx;
    int nBinsY, nBinsZ;
    
    nBinsY = binsN->y - bins0->y;
    nBinsZ = binsN->z - bins0->z;
    
    //int m;
    for(m=0; m<nRays; m++){
        //forward_proj_ray_t_u(phantom, sino, ray, dims, bins0, binsN);
        
        for(l=0; l<ray->n; l++){
            x_idx = dims->y*dims->z*ray->X[l];
            
            for(j=imax(0,ray->Y[l] + bins0->y); j<imin(ray->Y[l] + binsN->y, dims->y); j++){
                y_idx = dims->z*j;
                
                for(k=imax(0,ray->Z[l] + bins0->z); k<imin(ray->Z[l] + binsN->z,dims->z); k++){
                    
                    sinoIdx = m*nBinsY*nBinsZ + nBinsZ*(j - bins0->y - ray->Y[l]) + (k - bins0->z - ray->Z[l]);

                    phantIdx = x_idx + y_idx + k;
                    sino[2*sinoIdx] += phantom[phantIdx]*ray->L[l];
                    
                    phantIdx = dims->y*dims->z*j +  dims->z*ray->X[l] + k;
                    sino[2*sinoIdx+1] += phantom[phantIdx]*ray->L[l];
                }
            }
        }
        
        ray++;
    }
}




/*
int* unravel(int idx, int X, int Y, int Z)
{
    int idx3[3];

    idx3[2] = idx/Y/Z;
    idx3[1] = idx - idx3[2]*Y*Z;
    idx3[0] = idx % Z;

    return idx3;
}   
*/
