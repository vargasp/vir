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
VECT source:       A vect structure defined in the siddons.h that that contains the source location
VECT target:       A vect structure defined in the siddons.h that that contains the target location
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

void siddons(VECT source, VECT target, float *x_plane, float *y_plane, float *z_plane, float *x_alpha,
float *y_alpha, float *z_alpha, float *all_alpha, int X, int Y, int Z, float x_size, float y_size, float z_size, WEIGHTING *row)
{
  float delta_x, delta_y, delta_z, distance;
  float alpha_x1, alpha_xP, alpha_y1, alpha_yP, alpha_z1, alpha_zP;
  float alpha_min, alpha_max, delta_alpha, alpha_sum, alpha;
  int   i_min, i_max, j_min, j_max, k_min, k_max;
  int   i, j, k, index, a_index, x_index, y_index, z_index;
  float m_min[4], m_max[4];

  //Calculate the parametric values of the intersections of the line in question
  //with the first and last grid lines, both horizontal and vertical.
  delta_x  = target.x - source.x;
  delta_y  = target.y - source.y;
  delta_z  = target.z - source.z;
  distance = sqrt((delta_x*delta_x) + (delta_y*delta_y) + (delta_z*delta_z));

  if(delta_x != 0.0){
    alpha_x1 = (x_plane[0]-source.x)/delta_x;
    alpha_xP = (x_plane[X]-source.x)/delta_x;
  }else{
    alpha_x1 = 0.0;
    alpha_xP = 0.0;
  }

  if(delta_y != 0.0){
    alpha_y1 = (y_plane[0]-source.y)/delta_y;
    alpha_yP = (y_plane[Y]-source.y)/delta_y;
  }else{
    alpha_y1 = 0.0;
    alpha_yP = 0.0;
  }

  if(delta_z != 0.0){
    alpha_z1 = (z_plane[0]-source.z)/delta_z;
    alpha_zP = (z_plane[Z]-source.z)/delta_z;
  }else{
    alpha_z1 = 0.0;
    alpha_zP = 0.0;
  }

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

  //If alpha_max <= alpha_min, then the ray doesn't pass through the grid.
  if(alpha_max <= alpha_min) return;

  //Determine i_min, i_max, j_min, j_max, the indices of the first and last x and y planes
  //crossed by the line after entering the grid. We use ceil for mins and floor for maxs.
  if(delta_x > 0.0){
    i_min = floor(X - (x_plane[X] - alpha_min*delta_x - source.x)/x_size);
    i_max = ceil((source.x + alpha_max*delta_x - x_plane[0])/x_size - 1);
  }else{
    i_min = floor(X - (x_plane[X] - alpha_max*delta_x - source.x)/x_size);
    i_max = ceil((source.x + alpha_min*delta_x - x_plane[0])/x_size - 1);
  }

  if(delta_y > 0.0){
    j_min = floor(Y - (y_plane[Y] - alpha_min*delta_y - source.y)/y_size);
    j_max = ceil((source.y + alpha_max*delta_y - y_plane[0])/y_size - 1);
  }else{
    j_min = floor(Y - (y_plane[Y] - alpha_max*delta_y - source.y)/y_size);
    j_max = ceil((source.y + alpha_min*delta_y - y_plane[0])/y_size - 1);
  }

  if(delta_z > 0.0){
    k_min = floor(Z - (z_plane[Z] - alpha_min*delta_z - source.z)/z_size);
    k_max = ceil((source.z + alpha_max*delta_z - z_plane[0])/z_size - 1);
  }else{
    k_min = floor(Z - (z_plane[Z] - alpha_max*delta_z - source.z)/z_size);
    k_max = ceil((source.z + alpha_min*delta_z - z_plane[0])/z_size - 1);
  }

  //Compute the alpha values of the intersections of the line with all the relevant x planes in the grid.
  index = 0;
  if(delta_x > 0.0){
    for(i=i_min; i<= i_max; i++){
      x_alpha[index] = (x_plane[i]-source.x)/delta_x;
      ++index;
    }
  }else if(delta_x < 0.0){
    for(i=i_max; i>=i_min; i--){
       x_alpha[index] = (x_plane[i]-source.x)/delta_x;
       ++index;
    }
  }
  x_alpha[index] = 2.0;     /*Arbitrary flag to indicate the end of the sequence*/

  //Compute the alpha values of the intersections of the line with all the relevant y planes in the grid.
  index = 0;
  if(delta_y > 0.0){
    for(j=j_min; j<=j_max; j++){
      y_alpha[index] = (y_plane[j]-source.y)/delta_y;
      ++index;
    }
  }else if(delta_y < 0.0){
    for(j=j_max; j>=j_min; j--){
      y_alpha[index] = (y_plane[j]-source.y)/delta_y;
      ++index;
    }
  }
  y_alpha[index] = 2.0; /*Arbitrary flag to indicate the end of the sequence*/

  //Compute the alpha values of the intersections of the line with all the relevant z planes in the grid.
  index = 0;
  if(delta_z > 0.0){
    for(k=k_min; k<=k_max; k++){
      z_alpha[index] = (z_plane[k]-source.z)/delta_z;
      ++index;
    }
  }else if (delta_z < 0.0){
    for(k=k_max; k>=k_min; k--){
      z_alpha[index] = (z_plane[k]-source.z)/delta_z;
      ++index;
    }
  }
  z_alpha[index] = 2.0; /*Arbitrary flag to indicate the end of the sequence*/

  //Merges and sorts the alphas
  merge(x_alpha, y_alpha, z_alpha, alpha_min, alpha_max, all_alpha);

  //Loops through the alphas and calculates pixel length and pixel index
  index   = 0;
  a_index = 0;
  while(all_alpha[a_index+1] <= 1.0){
    delta_alpha = all_alpha[a_index+1]-all_alpha[a_index];
    if(delta_alpha > 0.0){
      alpha_sum = 0.5 * (all_alpha[a_index+1] + all_alpha[a_index]);
      x_index = (int)((source.x + alpha_sum*delta_x - x_plane[0])/x_size);
      y_index = (int)((source.y + alpha_sum*delta_y - y_plane[0])/y_size);
      z_index = (int)((source.z + alpha_sum*delta_z - z_plane[0])/z_size);
      row[index].pixel  = (X*Y*z_index) + (X*y_index) + (x_index);
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


void calc_grid_lines(float *x_lines, float *y_lines, float *z_lines, int X, int Y, int Z, float x_size, float y_size, float z_size)
{
  int i;

  x_lines[0] = -X*x_size/2.;
  y_lines[0] = -Y*y_size/2.;
  z_lines[0] = -Z*z_size/2.;

  for(i=1; i<X+1; i++) x_lines[i] = x_lines[i-1]+x_size;
  for(i=1; i<Y+1; i++) y_lines[i] = y_lines[i-1]+y_size;
  for(i=1; i<Z+1; i++) z_lines[i] = z_lines[i-1]+z_size;
}

