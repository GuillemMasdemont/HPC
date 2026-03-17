// demonstration of false sharing
// gcc -fopenmp falsesharing.c -o falsesharing
// sbatch falsesharing.sh

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

int main(int argc, char *argv[]) {
    double *counter;       
    int n = 2000000000;
    int dist;
    double dt;

    dist = atoi(argv[1]);
    counter = (double *)calloc(sizeof(double), dist+1);

    omp_set_num_threads(2);
    dt = omp_get_wtime();    
    #pragma omp parallel
    {
        int idx = omp_get_thread_num()*dist;
        for(int i = 0; i < n; i++)
            counter[idx]++;
    }
    dt = omp_get_wtime() - dt;
    
    printf("equal = %d\n", counter[0] == counter[dist]);
    printf("time = %lf\n\n", dt);

    free(counter);

    return 0;
}
