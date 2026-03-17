// Computation of pi using the Leibniz formula
// gcc -fopenmp -o pil6 pil6.c
// srun --cpus-per-task=2 pil6

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#define N 100000000

int main(void) {
	double pi = 0.0;
	double *mysum = NULL;

	#pragma omp parallel master
	mysum = (double *)malloc(omp_get_num_threads() * sizeof(double));

	double startTime = omp_get_wtime();
	#pragma omp parallel
	{
		double sum = 0.0;
		#pragma omp for
		for (int i = 0; i < N; i++) {
			int factor = 1 - 2 * (i % 2);
			sum += 4.0 * factor / (2 * i + 1);
		}
		mysum[omp_get_thread_num()] = sum;
	}

	#pragma omp parallel master
	{
		for (int i = 0; i < omp_get_num_threads(); i++)
			pi += mysum[i];
		free(mysum);
	}

	double endTime = omp_get_wtime();
	printf("pi: %lf, time taken: %lf seconds\n", pi, endTime - startTime);

	return 0;
}
