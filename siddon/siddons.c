/*
NAME:
siddons.c

PURPOSE:
An implementation of Siddon's algorithm (Med. Phys., 12, 252-255) for computing the intersection lengths of a line specified by the coordinates "source" and "target" with a (X,Y,Z) grid of pixels of dimension (z_size, y_size, z_size).

CATEGORY:
Ray Tracing

COMPILATION
gcc -c -Wall -Werror -fpic siddons.c
gcc -shared -o siddons.so siddons.o

CALLING SEQUENCE:
siddons(source, target, x_lines, y_lines, z_lines, x_alpha, y_alpha, z_alpha, all_alpha, X, Y, Z, x_size, y_size, z_size, weights);

INPUTS:
VECT src:       A vect structure defined in the siddons.h that that contains the source location
VECT trg:       A vect structure defined in the siddons.h that that contains the target location
int X:             The number of voxels in the grid in the x dimension 
int Y:             The number of voxels in the grid in the y dimension 
int Z:             The number of voxels in the grid in the z dimension 
float x_size:      The size of a voxel in the x dimension
float y_size:      The size of a voxel in the y dimension
float z_size:      The size of a voxel in the z dimension
float *x_plane:    A float array of of X+1 elements of the coordinates of the X grid lines
float *y_plane:    A float array of of Y+1 elements of the coordinates of the Y grid lines
float *z_plane:    A float array of of Z+1 elements of the coordinates of the Z grid lines
float *x_alpha:    A float array of of X+1 elements used to contain the set of parametric values
                     corresponding to the intersection of the ray and the x lines
float *y_alpha:    A float array of of Y+1 elements used to contain the set of parametric values
                     corresponding to the intersection of the ray and the y lines
float *z_alpha:    A float array of of Z+1 elements used to contain the set of parametric values
                     corresponding to the intersection of the ray and the z lines
float *all_alpha:  A float array of of 2*(max(X,Y,Z)+2)+1 elements used to contain the full set of parametric values
WEIGHTING *row:    An array of weighting stuctures containg the voxel index and weight

SIDE EFFECTS:
None.

RESTRICTIONS:
None

PROCEDURES:

MODIFICATION HISTORY:
Generalized and Optimized by Phillip Vargas, June 2009
Based on inters.c by Patrick La Riviere
*/
#include <math.h>
#include "siddons.h"
#include <stdio.h>
#include <stdlib.h>


float max4(float *array)
{
  float max;
  int   i;
  max = array[0];

  for(i=1; i<3; i++){
    if (array[i] > max) max = array[i];
  }

  return(max);
}


float min4(float *array)
{
  float min;
  int   i;
  min = array[0];

  for (i=1; i<3; i++){
    if (array[i] < min) min = array[i];
  }

  return(min);
}


void calc_grid_lines(float *x_lines, float *y_lines, float *z_lines, VectI nPixels, VectF dPixels)
{
  printf("dPixels.x,dPixels.y,dPixels.z: %f, %f, %f\n",dPixels.x,dPixels.y,dPixels.z);
  int i;

  x_lines[0] = -nPixels.x*dPixels.x/2.;
  y_lines[0] = -nPixels.y*dPixels.y/2.;
  z_lines[0] = -nPixels.z*dPixels.z/2.;

  for(i=1; i<nPixels.x+1; i++) x_lines[i] = x_lines[i-1]+dPixels.x;
  for(i=1; i<nPixels.y+1; i++) y_lines[i] = y_lines[i-1]+dPixels.y;
  for(i=1; i<nPixels.z+1; i++) z_lines[i] = z_lines[i-1]+dPixels.z;
  printf("dPixels.x,dPixels.y,dPixels.z: %f, %f, %f\n",dPixels.x,dPixels.y,dPixels.z);
  
}


void calc_alpha_range(float *m_min, float *m_max, float *plane, int N, float src, float dST, int idx)
{
  float alpha0, alphaN;

  if(dST != 0.0){
    alpha0 = (plane[0]-src)/dST;
    alphaN = (plane[N]-src)/dST;
  }else{
    alpha0 = 0.0;
    alphaN = 0.0;
  }
  
  if(alpha0 < alphaN){
    m_min[idx] = alpha0;
    m_max[idx] = alphaN;
  }else if(alpha0 > alphaN){
    m_min[idx] = alphaN;
    m_max[idx] = alpha0;
  }else{
    m_min[idx] = 0.0;
    m_max[idx] = 1.0;
  }
}


void calc_alphas(float *alpha, float *plane, float alpha_min, float alpha_max, float src, float dST, int N, float dPixel)
{
  int idx_min, idx_max, i, idx=0;

  //Determine idx_min and idx_max the indices of the first and last planes
  //crossed by the line after entering the grid. We use ceil for mins and floor for maxs.
  if(dST > 0.0){
    idx_min = floor(N - (plane[N] - alpha_min*dST - src)/dPixel);
    idx_max = ceil((src + alpha_max*dST - plane[0])/dPixel - 1);
  }else{
    idx_min = floor(N - (plane[N] - alpha_max*dST - src)/dPixel);
    idx_max = ceil((src + alpha_min*dST - plane[0])/dPixel - 1);
  }

  //Compute the alpha values of the intersections of the line with all the relevant x planes in the grid.
  if(dST > 0.0){
    for(i=idx_min; i<= idx_max; i++){
      alpha[idx] = (plane[i]-src)/dST;
      ++idx;
    }
  }else if(dST < 0.0){
    for(i=idx_max; i>=idx_min; i--){
       alpha[idx] = (plane[i]-src)/dST;
       ++idx;
    }
  }
  alpha[idx] = 2.0;     /*Arbitrary flag to indicate the end of the sequence*/
}


void merge(float *x_alpha, float *y_alpha, float *z_alpha, float min_alpha, float max_alpha, float *all_alpha)
{
  int all_index = 1;
  int x_index = 0;
  int y_index = 0;
  int z_index = 0;

  all_alpha[0] = min_alpha;

  while(x_alpha[x_index] <= 1.0 || y_alpha[y_index] <= 1.0 || z_alpha[z_index] <= 1.0){
    if((x_alpha[x_index] < y_alpha[y_index]) && (x_alpha[x_index] < z_alpha[z_index])){
      all_alpha[all_index] = x_alpha[x_index];
      ++x_index;
    }else if(y_alpha[y_index] < z_alpha[z_index]){
      all_alpha[all_index] = y_alpha[y_index];
      ++y_index;
    }else{
      all_alpha[all_index] = z_alpha[z_index];
      ++z_index;
    }
    ++all_index;
  }

  all_alpha[all_index] = max_alpha;
  all_alpha[all_index+1] = 2.0; /*an arbitrary flag to signal end of list.*/
}


//Loops through the alphas and calculates pixel length and pixel index
void calc_inter(VoxelLength *row, float *all_alpha, VectF src, 
                VectF dST, VectI nPixels, VectF dPixels,
                float *x_plane, float *y_plane, float *z_plane)
{
  float delta_alpha, alpha_sum, distance;
  int   index, a_index, x_index, y_index, z_index;
  
  distance = sqrt((dST.x*dST.x) + (dST.y*dST.y) + (dST.z*dST.z));

  index   = 0;
  a_index = 0;
  while(all_alpha[a_index+1] <= 1.0){
    delta_alpha = all_alpha[a_index+1]-all_alpha[a_index];
    if(delta_alpha > 0.0){
      printf("delta_alpha %f\n", delta_alpha);
      alpha_sum = 0.5 * (all_alpha[a_index+1] + all_alpha[a_index]);
      x_index = (int)((src.x + alpha_sum*dST.x - x_plane[0])/dPixels.x);
      y_index = (int)((src.y + alpha_sum*dST.y - y_plane[0])/dPixels.y);
      z_index = (int)((src.z + alpha_sum*dST.z - z_plane[0])/dPixels.z);
      
      printf("x_index,y_index,z_index: %d, %d, %d\n",x_index,y_index,z_index);
      printf("nPixels.x,nPixels.y,nPixels.z: %d, %d, %d\n",nPixels.x,nPixels.y,nPixels.z);
      printf("dPixels.x,dPixels.y,dPixels.z: %f, %f, %f\n",dPixels.x,dPixels.y,dPixels.z);

      printf("t: %d",  (nPixels.x*nPixels.y*z_index) + (nPixels.x*y_index) + (x_index));
      row[index].idx  = (nPixels.x*nPixels.y*z_index) + (nPixels.x*y_index) + (x_index);
      row[index].length = distance*delta_alpha;
      printf("Idx: %d\n",row[index].idx);
      printf("length: %f\n",row[index].length);
      
      ++index;
    }
    ++a_index;
  }
  row[index].idx  = -1; /*Flag denoting end of file*/
}


void siddons(VectF src, VectF trg, VectI nPixels, VectF dPixels, float *x_plane, float *y_plane, float *z_plane, float *x_alpha,
             float *y_alpha, float *z_alpha, float *all_alpha,
             VoxelLength *row)
{

  printf("Size of Int: %lu\n", sizeof(int));
  printf("Size of Float: %lu\n", sizeof(float));
  printf("Here 1\n");
  printf("src.x,src.y,src.z: %f, %f, %f\n",src.x,src.y,src.z);
  printf("trg.x,trg.y,trg.z: %f, %f, %f\n",trg.x,trg.y,trg.z);
  printf("nPixels.x,nPixels.y,nPixels.z: %d, %d, %d\n",nPixels.x,nPixels.y,nPixels.z);
  printf("dPixels.x,dPixels.y,dPixels.z: %f, %f, %f\n",dPixels.x,dPixels.y,dPixels.z);
  
  
  
  
  VectF dST;
  float alpha_min, alpha_max;
  float m_min[4], m_max[4];

  //Calculate the parametric values of the intersections of the line in question
  //with the first and last grid lines, both horizontal and vertical.
  dST.x  = trg.x - src.x;
  dST.y  = trg.y - src.y;
  dST.z  = trg.z - src.z;

  //Calculate alpha_min and alpha_max, which is either the parametric value of the intersection
  // where the line of interest enters the grid, or 0.0 if the source is inside the grid.
  m_min[0] = 0.0;
  m_max[0] = 1.0;

  //Calculate the alpha ranges across planes
  calc_alpha_range(m_min, m_max, x_plane, nPixels.x, src.x, dST.x, 1);
  calc_alpha_range(m_min, m_max, y_plane, nPixels.y, src.y, dST.y, 2);
  calc_alpha_range(m_min, m_max, z_plane, nPixels.z, src.z, dST.z, 3);
  printf("Here 2\n");

  alpha_min = max4(m_min);
  alpha_max = min4(m_max);

  //If alpha_max <= alpha_min, then the ray doesn't pass through the grid.
  if(alpha_max <= alpha_min) return;
  
  //Compute the alpha values of the intersections of the line with all the relevant x planes in the grid.
  calc_alphas(x_alpha, x_plane, alpha_min, alpha_max, src.x,  dST.x,  nPixels.x,  dPixels.x);
  calc_alphas(y_alpha, y_plane, alpha_min, alpha_max, src.y,  dST.y,  nPixels.y,  dPixels.y);
  calc_alphas(z_alpha, z_plane, alpha_min, alpha_max, src.z,  dST.z,  nPixels.z,  dPixels.z);
  printf("Here 3\n");

  //Merges and sorts the alphas
  merge(x_alpha, y_alpha, z_alpha, alpha_min, alpha_max, all_alpha);
  printf("Here 4\n");

  calc_inter(row, all_alpha, src, dST, nPixels, dPixels,
             x_plane, y_plane, z_plane);
  printf("Here 5\n");

}

