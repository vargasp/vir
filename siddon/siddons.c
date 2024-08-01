/*
NAME:
siddons.c

PURPOSE:
An implementation of Siddon's algorithm (Med. Phys., 12, 252-255) for computing the intersection lengths of a line specified by the coordinates "source" and "target" with a (X,Y,Z) grid of pixels of dimension (z_size, y_size, z_size).

CATEGORY:
Ray Tracing

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


void calc_alpha_range(float *alpha0, float *alphaN, float plane0, float planeN, float src, float dST);
void calc_idx_range(int *idx_min, int *idx_max, float plane0, float planeN, float alpha_min, float alpha_max, float src, float dST, int nPixel, float dPixel);


void calc_alpha_range(float *alpha0, float *alphaN, float plane0, float planeN, float src, float dST)
{
  if(dST != 0.0){
    *alpha0 = (plane0-src)/dST;
    *alphaN = (planeN-src)/dST;
  }else{
    *alpha0 = 0.0;
    *alphaN = 0.0;
  }
}


void calc_idx_range(int *idx_min, int *idx_max, float plane0, float planeN, float alpha_min, float alpha_max, float src, float dST, int nPixel, float dPixel)
{
  if(dST > 0.0){
    *idx_min = floor(nPixel - (planeN - alpha_min*dST - src)/dPixel);
    *idx_max = ceil((src + alpha_max*dST - plane0)/dPixel - 1);
  }else{
    *idx_min = floor(nPixel - (planeN - alpha_max*dST - src)/dPixel);
    *idx_max = ceil((src + alpha_min*dST - plane0)/dPixel - 1);
  }
}


void siddons(VECT src, VECT trg, float *x_plane, float *y_plane, float *z_plane, float *x_alpha,
float *y_alpha, float *z_alpha, float *all_alpha, int nX, int nY, int nZ, float dX, float dY, float dZ, WEIGHTING *row)
{
  float dSTx, dSTy, dSTz, distance;
  float alpha_x1, alpha_xP, alpha_y1, alpha_yP, alpha_z1, alpha_zP;
  float alpha_min, alpha_max, delta_alpha, alpha_sum;
  int   i_min, i_max, j_min, j_max, k_min, k_max;
  int   i, j, k, index, a_index, x_index, y_index, z_index;
  float m_min[4], m_max[4];

  //Calculate the parametric values of the intersections of the line in question
  //with the first and last grid lines, both horizontal and vertical.
  dSTx  = trg.x - src.x;
  dSTy  = trg.y - src.y;
  dSTz  = trg.z - src.z;
  distance = sqrt((dSTx*dSTx) + (dSTy*dSTy) + (dSTz*dSTz));


  printf("Src x: %f, y: %f, z: %f\n", src.x,src.y, src.z);
  printf("Trg x: %f, y: %f, z: %f\n", trg.x,trg.y, trg.z);
  printf("Distance: %f\n", distance);
  
  
  calc_alpha_range(&alpha_x1, &alpha_xP, x_plane[0], x_plane[nX], src.x, dSTx);
  calc_alpha_range(&alpha_y1, &alpha_yP, y_plane[0], y_plane[nY], src.y, dSTy);
  calc_alpha_range(&alpha_z1, &alpha_zP, z_plane[0], z_plane[nZ], src.z, dSTz);
  
  /*
  if(dSTx != 0.0){
    alpha_x1 = (x_plane[0]-src.x)/dSTx;
    alpha_xP = (x_plane[nX]-src.x)/dSTx;
  }else{
    alpha_x1 = 0.0;
    alpha_xP = 0.0;
  }
    
  if(dSTy != 0.0){
    alpha_y1 = (y_plane[0]-src.y)/dSTy;
    alpha_yP = (y_plane[nY]-src.y)/dSTy;
  }else{
    alpha_y1 = 0.0;
    alpha_yP = 0.0;
  }

  if(dSTz != 0.0){
    alpha_z1 = (z_plane[0]-src.z)/dSTz;
    alpha_zP = (z_plane[nZ]-src.z)/dSTz;
  }else{
    alpha_z1 = 0.0;
    alpha_zP = 0.0;
  }

  */

  printf("Alphas X: %f, %f\n", alpha_x1, alpha_xP);

  //Calculate alpha_min, which is either the parametric value of the intersection
  // where the line of interest enters the grid, or 0.0 if the source is inside the grid.
  //Calculate alpha_max, which is either the parametric value of the intersection
  // where the line of interest leaves the grid, or 1.0 if the target is inside the grid.
  m_min[0] = 0.0;
  m_max[0] = 1.0;

  if(alpha_x1 < alpha_xP){
    m_min[1] = alpha_x1;
    m_max[1] = alpha_xP;
  }else if(alpha_x1 > alpha_xP){
    m_min[1] = alpha_xP;
    m_max[1] = alpha_x1;
  }else{
    m_min[1] = 0.0;
    m_max[1] = 1.0;
  }

  if(alpha_y1 < alpha_yP){
    m_min[2] = alpha_y1;
    m_max[2] = alpha_yP;
  }else if(alpha_y1 > alpha_yP){
    m_min[2] = alpha_yP;
    m_max[2] = alpha_y1;
  }else{
    m_min[2] = 0.0;
    m_max[2] = 1.0;
  }
  
  if(alpha_z1 < alpha_zP){
    m_min[3] = alpha_z1;
    m_max[3] = alpha_zP;
  }else if(alpha_z1 > alpha_zP){
    m_min[3] = alpha_zP;
    m_max[3] = alpha_z1;
  }else{
    m_min[3] = 0.0;
    m_max[3] = 1.0;
  }

  alpha_min = max4(m_min);
  alpha_max = min4(m_max);


  printf("Alphas Min/Max: %f, %f\n",alpha_min, alpha_max);


  //If alpha_max <= alpha_min, then the ray doesn't pass through the grid.
  if(alpha_max <= alpha_min) return;
  
  
  
  //Determine i_min, i_max, j_min, j_max, the indices of the first and last x and y planes
  //crossed by the line after entering the grid. We use ceil for mins and floor for maxs.
  
  calc_idx_range(&i_min, &i_max, x_plane[0], x_plane[nX], alpha_min, alpha_max, src.x, dSTx, nX, dX);
  calc_idx_range(&j_min, &j_max, y_plane[0], y_plane[nY], alpha_min, alpha_max, src.y, dSTy, nY, dY);
  calc_idx_range(&k_min, &k_max, z_plane[0], z_plane[nZ], alpha_min, alpha_max, src.z, dSTz, nZ, dZ);

  /*  
  if(dSTx > 0.0){
    i_min = floor(nX - (x_plane[nX] - alpha_min*dSTx - src.x)/dX);
    i_max = ceil((src.x + alpha_max*dSTx - x_plane[0])/dX - 1);
  }else{
    i_min = floor(nX - (x_plane[nX] - alpha_max*dSTx - src.x)/dX);
    i_max = ceil((src.x + alpha_min*dSTx - x_plane[0])/dX - 1);
  }

  if(dSTy > 0.0){
    j_min = floor(nY - (y_plane[nY] - alpha_min*dSTy - src.y)/dY);
    j_max = ceil((src.y + alpha_max*dSTy - y_plane[0])/dY - 1);
  }else{
    j_min = floor(nY - (y_plane[nY] - alpha_max*dSTy - src.y)/dY);
    j_max = ceil((src.y + alpha_min*dSTy - y_plane[0])/dY - 1);
  }

  if(dSTz > 0.0){
    k_min = floor(nZ - (z_plane[nZ] - alpha_min*dSTz - src.z)/dY);
    k_max = ceil((src.z + alpha_max*dSTz - z_plane[0])/dY - 1);
  }else{
    k_min = floor(nZ - (z_plane[nZ] - alpha_max*dSTz - src.z)/dY);
    k_max = ceil((src.z + alpha_min*dSTz - z_plane[0])/dY - 1);
  }
  */

  printf("i_min: %d, i_max %d:\n", i_min, i_max);
  printf("X_0: %f, X_N %f:\n", x_plane[0], x_plane[nX]);


  //Compute the alpha values of the intersections of the line with all the relevant x planes in the grid.
  index = 0;
  if(dSTx > 0.0){
    for(i=i_min; i<= i_max; i++){
      x_alpha[index] = (x_plane[i]-src.x)/dSTx;
      ++index;
    }
  }else if(dSTx < 0.0){
    for(i=i_max; i>=i_min; i--){
       x_alpha[index] = (x_plane[i]-src.x)/dSTx;
       ++index;
    }
  }
  x_alpha[index] = 2.0;     /*Arbitrary flag to indicate the end of the sequence*/

  //Compute the alpha values of the intersections of the line with all the relevant y planes in the grid.
  index = 0;
  if(dSTy > 0.0){
    for(j=j_min; j<=j_max; j++){
      y_alpha[index] = (y_plane[j]-src.y)/dSTy;
      ++index;
    }
  }else if(dSTy < 0.0){
    for(j=j_max; j>=j_min; j--){
      y_alpha[index] = (y_plane[j]-src.y)/dSTy;
      ++index;
    }
  }
  y_alpha[index] = 2.0; /*Arbitrary flag to indicate the end of the sequence*/

  //Compute the alpha values of the intersections of the line with all the relevant z planes in the grid.
  index = 0;
  if(dSTz > 0.0){
    for(k=k_min; k<=k_max; k++){
      z_alpha[index] = (z_plane[k]-src.z)/dSTz;
      ++index;
    }
  }else if (dSTz < 0.0){
    for(k=k_max; k>=k_min; k--){
      z_alpha[index] = (z_plane[k]-src.z)/dSTz;
      ++index;
    }
  }
  z_alpha[index] = 2.0; /*Arbitrary flag to indicate the end of the sequence*/


  printf("D\n");
  //Merges and sorts the alphas
  merge(x_alpha, y_alpha, z_alpha, alpha_min, alpha_max, all_alpha);

  printf("E\n");

  //Loops through the alphas and calculates pixel length and pixel index
  index   = 0;
  a_index = 0;
  while(all_alpha[a_index+1] <= 1.0){
    delta_alpha = all_alpha[a_index+1]-all_alpha[a_index];
    if(delta_alpha > 0.0){
      printf("delta_alpha: %f\n", delta_alpha);
      alpha_sum = 0.5 * (all_alpha[a_index+1] + all_alpha[a_index]);
      x_index = (int)((src.x + alpha_sum*dSTx - x_plane[0])/dX);
      y_index = (int)((src.y + alpha_sum*dSTy - y_plane[0])/dY);
      z_index = (int)((src.z + alpha_sum*dSTz - z_plane[0])/dZ);
      row[index].pixel  = (nX*nY*z_index) + (nX*y_index) + (x_index);
      row[index].length = distance*delta_alpha;
      ++index;
    }
    ++a_index;
  }
  row[index].pixel  = -1; /*Flag denoting end of file*/
}


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


void calc_grid_lines(float *x_lines, float *y_lines, float *z_lines, int nX, int nY, int nZ, float dX, float dY, float dZ)
{
  int i;

  x_lines[0] = -nX*dX/2.;
  y_lines[0] = -nY*dY/2.;
  z_lines[0] = -nZ*dZ/2.;

  for(i=1; i<nX+1; i++) x_lines[i] = x_lines[i-1]+dX;
  for(i=1; i<nY+1; i++) y_lines[i] = y_lines[i-1]+dY;
  for(i=1; i<nZ+1; i++) z_lines[i] = z_lines[i-1]+dZ;
}

