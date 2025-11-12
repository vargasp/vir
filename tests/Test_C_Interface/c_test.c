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
void Func2(int i_test, float f_test, double d_test);
void Func3(int *i_test, float *f_test, double *d_test);
void Func4i(int n, int *i_test_array);
void Func4f(int n, float *f_test_array);
void Func4d(int n, double *d_test_array);
void Func5(int N, int M, float *f_test_array);
void Func6(DATA *data);
void Func8(Ray *ray);
void Func9(Ray *ray, int nRays);


/*
This is a function to test printing/stdio in C being called by a python wrapper
*/
void Func0()
{
    printf("Function 0\n");
    printf("Hello\n\n");
}


/*
This is a function to confirm bitsize of and int, float, and double in C being
 called by a python wrapper
*/
void Func1()
{
    int a = 10;
    float b = 10.0;
    double c = 10.0;
    
    printf("Function 1\n");
    printf("Size of in int: %zu bits\n", 8*sizeof(a));
    printf("Size of in float: %zu bits\n", 8*sizeof(b));
    printf("Size of in double: %zu bits\n\n", 8*sizeof(c));
}


/*
This is a function to confirm passing ints, floats, and doubles to C by a pyhton
wrapper
*/
void Func2(int i_test, float f_test, double d_test)
{
    printf("Function 2\n");
    printf("This is an int/float/double scalar test.\n");
    printf("The passed integer: %d\n", i_test);
    printf("The passed float: %0.1f\n", f_test);
    printf("The passed double: %0.1f\n\n", d_test);
}


/*
This is a  function to confirm passing pointers to C by a pyhton wrapper
*/
void Func3(int *i_test, float *f_test, double *d_test)
{
    printf("Function 3\n");
    printf("This is an int/float/double scalar test.\n");
    printf("The passed integer by pointer: %d\n", *i_test);
    printf("The passed float by pointer: %0.1f\n", *f_test);
    printf("The passed double by pointer: %0.1f\n\n", *d_test);
    
    /*Modify the values*/
    *i_test = 10;
    *f_test = 20.0;
    *d_test = 30.0;
}


/*
This is a function to confirm passing arrays to C by a pyhton wrapper
*/
void Func4i(int n, int *i_test_array)
{
    int i;
    
    printf("Function 4i\n");
    printf("The passed int array:\n");
     for (i=0; i<n; i++){
         printf("%d ", i_test_array[i]);
     }
    printf("\n\n");
    
    /*Modify the values*/
    i_test_array[0] = 10;
}

void Func4f(int n, float *f_test_array)
{
    int i;
    
    printf("Function 4f\n");
    printf("The passed float array:\n");
     for (i=0; i<n; i++){
         printf("%0.1f ", f_test_array[i]);
     }
    printf("\n\n");
    
    /*Modify the values*/
    f_test_array[0] = 10.0;   
}

void Func4d(int n, double *d_test_array)
{
    int i;
    
    printf("Function 4d\n");
    printf("The passed double array:\n");
     for (i=0; i<n; i++){
         printf("%0.1f ", d_test_array[i]);
     }
    printf("\n\n");
    
    /*Modify the values*/
    d_test_array[0] = 10.0;   
}

/*
This is a function to confirm passing 2d float arrays to C by a pyhton wrapper
*/
void Func5(int N, int M, float *f_test_array)
{
    int n,m;
    int p=0;
    
    printf("Function 5\n");
    printf("This is a 2d float array test.\n");
    printf("Outer axis (rows): %d, inner axis (cols): %d.\n",N,M);
    for (n=0; n<N; n++){
        for (m=0; m<M; m++){
            printf("outer_idx=%d, inner_idx=%d, flat_idx=%d, elem=%.1f\n", n,m,p, f_test_array[p]);
            p++;
        }
     }
    printf("\n");
}


void Func6(DATA *data)
{
    int i;

    printf("Function 6\n");
    printf("This is a structure test.\n");
    
    printf("The passed double input array:\n");
    for (i=0; i<data->n; i++){
         printf("%.1f ", data->ina[i]);
    }
    printf("\n");

    printf("The passed double output array:\n");
    for (i=0; i<data->n; i++){
         printf("%.1f ", data->outa[i]);
    }
    printf("\n\n");

    /*Modify the output array*/
    for (i=0; i<data->n; i++){
        data->outa[i] = data->ina[i]*data->ina[i];
    }
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






