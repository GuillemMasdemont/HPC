// Computation of pi using the Leibniz formula
// gcc -fopenmp -o pil1 pil1.c
// srun --cpus-per-task=2 pil1

#include <stdio.h>
#include "omp.h"

#define N 1000000

int main(void) {
	double pi = 0.0;

	int factor = 1;
	#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		pi += 4.0 * factor / (2 * i + 1);
		factor = -factor;
	}

	printf("pi: %lf\n", pi);
	
	return 0;
}
