// srun --reservation=fri gcc -O2 -fopenmp -o abs abs.c
// srun --reservation=fri --threads-per-core=1 --mem=4G abs

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#define ALIGN 32

__attribute__ ((noinline)) int myAbs (int x) {       // force compiler not to inline a function
    return x < 0 ? -x : x; 
}

#pragma omp declare simd
__attribute__ ((noinline)) int myAbsSIMD (int x) {   // force compiler not to inline a function
    return x < 0 ? -x : x; 
}

int main(int argc, char *argv[]) {
    int *x;         // vector
    double dt;      // time difference

    // get arguments
    int n = 1000000000;

    // allocate and initialize vectors
    x = (int *)aligned_alloc(ALIGN, n * sizeof(int));
    
    // compute minimum
    dt = omp_get_wtime();
    #pragma omp simd
    for(int i = 0; i < n; i++)
        x[i] = myAbs(x[i]);
    dt = omp_get_wtime() - dt;
    printf("omp simd: %lf\n\n", dt);

    // compute minimum SIMD
    dt = omp_get_wtime();
    #pragma omp simd
    for(int i = 0; i < n; i++)
        x[i] = myAbsSIMD(x[i]);
    dt = omp_get_wtime() - dt;
    printf("omp simd: %lf\n\n", dt);

    // free vectors 
    free(x);

    return 0;
}
