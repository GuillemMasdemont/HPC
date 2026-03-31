// Computation of pi using the Leibniz formula
// gcc -fopenmp -o pil4 pil4.c
// srun --cpus-per-task=2 pil4

#include <stdio.h>
#include "omp.h"

#define N 100000000

omp_lock_t my_lock;

int main(void) {
	double pi = 0.0;
	omp_init_lock(&my_lock);

	double startTime = omp_get_wtime();
	#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		int factor = 1 - 2 * (i % 2);

		omp_set_lock(&my_lock);
		pi += 4.0 * factor / (2 * i + 1);			
		omp_unset_lock(&my_lock);
	}
	double endTime = omp_get_wtime();
	printf("pi: %lf, time taken: %lf seconds\n", pi, endTime - startTime);

	omp_destroy_lock(&my_lock);

	return 0;
}
