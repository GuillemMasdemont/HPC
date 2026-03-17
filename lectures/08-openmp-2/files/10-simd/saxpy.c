// check AVX compatibility: srun --reservation=fri lscpu | grep avx
//
// srun --reservation=fri gcc -fopenmp -o saxpy saxpy.c 
// srun --reservation=fri --ntasks=1 --cpus-per-task=16 --threads-per-core=1 --mem=48G saxpy 1600000000 16
//
// srun --reservation=fri gcc -O2 -fopenmp -o saxpy saxpy.c 
// srun --reservation=fri --ntasks=1 --cpus-per-task=16 --threads-per-core=1 --mem=48G saxpy 1600000000 16

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#define ALIGN 32

int main(int argc, char *argv[]) {
    double a;                       // scalar
    double *x, *y;                  // vectors
    double *xAligned, *yAligned;    // aligned vectors
    int n, p;                       // problem size, number of tasks
    double dt;                      // time difference    

    // get arguments
    n = atoi(argv[1]);
    p = atoi(argv[2]);

    // allocate and initialize vectors (sequential)
    x = (double *)malloc(n*sizeof(double));
    y = (double *)malloc(n*sizeof(double));
    xAligned = (double *)aligned_alloc(ALIGN, n * sizeof(double));
    yAligned = (double *)aligned_alloc(ALIGN, n * sizeof(double));

    // initialize
    a = 2.0;
    for (int i = 0; i < n; i++) {
        x[i] = 1.0;
        y[i] = 1.0;
        xAligned[i] = 1.0;
        yAligned[i] = 1.0;
    }

    omp_set_num_threads(p);

    // tiled map with sequential reduction of intermediate results
    dt = omp_get_wtime();
    for (int iter = 0; iter < 10; iter++)
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            y[i] = a * x[i] + y[i];        
    dt = omp_get_wtime() - dt;
    printf("omp: %lf\n\n", dt);

    // tiled map with sequential reduction of intermediate results
    dt = omp_get_wtime();
    for (int iter = 0; iter < 10; iter++)
        #pragma omp parallel for simd
        for (int i = 0; i < n; i++)
            y[i] = a * x[i] + y[i];        
    dt = omp_get_wtime() - dt;
    printf("omp+simd: %lf\n\n", dt);

    // tiled map with sequential reduction of intermediate results
    dt = omp_get_wtime();
    for (int iter = 0; iter < 10; iter++)
        #pragma omp parallel for simd aligned(xAligned, yAligned: ALIGN)
        for (int i = 0; i < n; i++)
            yAligned[i] = a * xAligned[i] + yAligned[i];        
    dt = omp_get_wtime() - dt;
    printf("omp+simd+aligned: %lf\n\n", dt);

    // free vectors 
    free(x);
    free(y);
    free(xAligned);
    free(yAligned);

    return 0;
}
