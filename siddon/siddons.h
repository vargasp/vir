typedef struct {
    int   pixel;
    float length;
} WEIGHTING;

typedef struct {
    float x;
    float y;
    float z;
} VECT;

float max4(float *array);
float min4(float *array);
void merge(float *x_alpha, float *y_alpha, float *z_alpha, float min_alpha, float max_alpha, float *all_alpha);
void calc_grid_lines(float *x_lines, float *y_lines, float *z_lines, int x, int y, int z, float x_size, float y_size, float z_size);
void siddons(VECT source, VECT target, float *x_plane, float *y_plane, float *z_plane, float *x_alpha,
  float *y_alpha, float *z_alpha, float *all_alpha, int X, int Y, int Z, float x_size, float y_size, float z_size, WEIGHTING *row);

