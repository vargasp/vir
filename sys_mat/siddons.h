typedef struct {
    int x;
    int y;
    int z;
} VectI;


typedef struct {
    float x;
    float y;
    float z;
} VectF;


typedef struct {
    int   idx;
    float length;
} VoxelLength;


float max4(float *array);
float min4(float *array);

void merge(float *x_alpha, float *y_alpha, float *z_alpha,
           float min_alpha, float max_alpha, float *all_alpha);
void calc_grid_lines(float *x_lines, float *y_lines, float *z_lines,
                     VectI nPixels, VectF dPixels);

void calc_alpha_range(float *alpha0, float *alphaN, float *plane0,
                      int N, float src, float dST, int idx);
void calc_alphas(float *alpha, float *plane, float alpha_min, float alpha_max,
                 float src, float dST, int N, float dPixel);

void calc_inter(VoxelLength *row, float *all_alpha, VectF src, 
                VectF dST, VectI nPixels, VectF dPixels,
                float *x_plane, float *y_plane, float *z_plane);

void sum_inter(float *ray_sum, float *iArray, float *all_alpha, VectF src, 
                VectF dST, VectI nPixels, VectF dPixels,
                float *x_plane, float *y_plane, float *z_plane);


void siddons(VectF src, VectF trg, VectI nPixels, VectF dPixels,
             float *x_plane, float *y_plane, float *z_plane,
             float *x_alpha, float *y_alpha, float *z_alpha, float *all_alpha,
             VoxelLength *row);

void siddons_proj(float *ray_sum, float *iArray, VectF src, VectF trg, VectI nPixels, VectF dPixels,
             float *x_plane, float *y_plane, float *z_plane, float *x_alpha,
             float *y_alpha, float *z_alpha, float *all_alpha);


void forward_project(float *iArray, float *dArray, VectI nPixels, VectF dPixels);