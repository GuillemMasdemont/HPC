// Computation of pi using the Leibniz formula
// gcc -fopenmp -o pil0 pil0.c
// srun --cpus-per-task=2 pil0

#include <stdio.h>
#include "omp.h"

#define N 1000000

int main(void) {
	double pi = 0.0;

	int factor = 1;
	#pragma omp parallel for
	for (int i = 0; i < N; i++, factor = -factor) {
		pi += 4.0 * factor / (2 * i + 1);
	}

	printf("pi: %lf\n", pi);
	return 0;
}
