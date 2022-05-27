//
//  projection.c
//  
//
//  Created by Phillip Vargas on 4/16/15.
//
//
// Compilation and linking:
// gcc -c -Wall -Werror -fpic c_test.c
// gcc -shared -o c_test.so c_test.o


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

//struct DATA;

typedef struct DATA{
    int n;
    double *ina;
    double *outa;
} DATA;


//struct Ray;

typedef struct Ray{
    int n;
    int *X;
    int *Y;
    int *Z;
    float *L;
} Ray;



void Func0();
void Func1();
void Func2(int i_test, float f_test);
void Func3(int i_test, int *i_test_array);
void Func4(int i_test, float *f_test_array);
void Func5(int N, int M, float *f_test_array);
void Func6(int idx, int N, float *f_test_array);
void Func7(DATA *data);
void Func8(Ray *ray);
void Func9(Ray *ray, int nRays);



void Func0()
{
    printf("Function 0\n");
    printf("Hello\n\n");
}


void Func1()
{
    int a = 10;
    float b = 10.0;
    double c = 10.0;
    
    printf("Function 1\n");
    printf("Size of in int: %zu\n", 8*sizeof(a));
    printf("Size of in float: %zu\n", 8*sizeof(b));
    printf("Size of in double: %zu\n\n", 8*sizeof(c));
}

void Func2(int i_test, float f_test)
{
    printf("Function 2\n");
    printf("This is an int/float scalar test.\n");
    printf("Print the passed integer: %d\n", i_test);
    printf("Print the passed float: %f\n\n", f_test);
}


void Func3(int i_test, int *i_test_array)
{
    int i;
    
    printf("Function 3\n");
    printf("This is an int array test.\n");
     for (i=0; i<i_test; i++){
         printf("%d ", i_test_array[i]);
     }
    printf("\n\n");
}

void Func4(int i_test, float *f_test_array)
{
    int i;
    
    printf("Function 4\n");
    printf("This is a float array test.\n");
     for (i=0; i<i_test; i++){
         printf("%.2f ", f_test_array[i]);
     }
    printf("\n\n");
}


void Func5(int N, int M, float *f_test_array)
{
    int n,m;
    int p=0;
    
    printf("Function 5\n");
    printf("This is a 2d float array test.\n");
    for (n=0; n<N; n++){
         for (m=0; m<M; m++){
             printf("outer_idx=%d, inner_idx=%d, flat_idx=%d, elem=%.2f\n", n,m,p, f_test_array[p]);
             p++;
         }
     }
    printf("\n\n");
}


void Func6(int idx, int N, float *f_array)
{
    int i;
    
    printf("Function 6\n");
    printf("This is a pass by refernce test.\n");
    
    f_array[idx] = 22.0;
    
    for (i=0; i<N; i++){
        printf("%.2f ", f_array[i]);
    }
    printf("\n\n");
}


void Func7(DATA *data)
{
    int i;

    printf("Function 7\n");
    printf("This is a structure test.\n");
    
    printf("Interger: %d\n", data->n);
    printf("Double Array.\n");
    for (i=0; i<data->n; i++){
        printf("%.2f ", data->ina[i]);
    }
    
    for (i=0; i<data->n; i++){
        data->outa[i] = data->ina[i]*data->ina[i];
        }
    printf("\n\n");
}



void Func8(Ray *ray)
{
    int i;

    printf("Function 8\n");
    printf("This is a structure array test.\n");
    
    printf("n: %d\n", ray->n);
    printf("X:");
    for (i=0; i<ray->n; i++){
        printf(" %d", ray->X[i]);
    }
    printf("\n"); 

    printf("Y:");
    for (i=0; i<ray->n; i++){
        printf(" %d", ray->Y[i]);
    }    
    printf("\n"); 

    printf("Z:");
    for (i=0; i<ray->n; i++){
        printf(" %d", ray->Z[i]);
    }
    printf("\n"); 
    
    for (i=0; i<ray->n; i++){
        printf("%.2f ", ray->L[i]);
    }
    
    printf("\n\n");
}

void Func9(Ray *ray, int nRays)
{
    int i, j;
    
    printf("Function 9\n");
    printf("This is a structure array test.\n");
    
    
    for (j=0; j<nRays; j++){

        printf("n: %d\n", ray->n);
        printf("X:");
        for (i=0; i<ray->n; i++){
            printf(" %d", ray->X[i]);
        }
        printf("\n");
    
        printf("Y:");
        for (i=0; i<ray->n; i++){
            printf(" %d", ray->Y[i]);
        }
        printf("\n");
    
        printf("Z:");
        for (i=0; i<ray->n; i++){
            printf(" %d", ray->Z[i]);
        }
        printf("\n");
        
        for (i=0; i<ray->n; i++){
            printf("%.2f ", ray->L[i]);
        }
        
        printf("\n\n");
        ray++;
    }
}


void Func10(int *i_test, float *f_test)
{
    *i_test +=1;
    f_test++;
    
    printf("Test: %d %f\n", *i_test, *f_test);
}






