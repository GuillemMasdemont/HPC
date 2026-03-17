// Computation of pi using the Leibniz formula
// gcc -fopenmp -o pil2 pil2.c
// srun --cpus-per-task=2 pil2

#include <stdio.h>
#include "omp.h"

#define N 100000000

int main(void) {
	double pi = 0.0;

	double startTime = omp_get_wtime();
	#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		int factor = 1 - 2 * (i % 2);
		#pragma omp atomic
		pi += 4.0 * factor / (2 * i + 1);
	}
	double endTime = omp_get_wtime();
	printf("pi: %lf, time taken: %lf seconds\n", pi, endTime - startTime);

	return 0;
}
