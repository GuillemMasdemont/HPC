// Computation of pi using the Leibniz formula
// gcc -fopenmp -o pil7 pil7.c
// srun --cpus-per-task=2 pil7

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#define N 100000000

int main(void) {
	double pi = 0.0;
	double *myPiPart = NULL;

	#pragma omp parallel master
	myPiPart = (double *)malloc(omp_get_num_threads() * sizeof(double));

	double startTime = omp_get_wtime();
	#pragma omp parallel
	{
		double sum = 0.0;
		#pragma omp for
		for (int i = 0; i < N; i++) {
			int factor = 1 - 2 * (i % 2);
			sum += 4.0 * factor / (2 * i + 1);
		}
		myPiPart[omp_get_thread_num()] = sum;
	}
	for (int i = 0; i < omp_get_num_threads(); i++) {
		pi += myPiPart[i];
	}
	double endTime = omp_get_wtime();

	free(myPiPart);
	printf("pi: %lf, time taken: %lf seconds\n", pi, endTime - startTime);

	return 0;
}
