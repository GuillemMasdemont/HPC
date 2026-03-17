// Computation of pi using the Leibniz formula
// gcc -fopenmp -o pil5 pil5.c
// srun --cpus-per-task=2 pil5

#include <stdio.h>
#include "omp.h"

#define N 100000000

int main(void) {
	double pi = 0.0;

	double startTime = omp_get_wtime();
	#pragma omp parallel
	{
		double local_pi = 0.0;
		#pragma omp for
		for (int i = 0; i < N; i++) {
			int factor = 1 - 2 * (i % 2);
			local_pi += 4.0 * factor / (2 * i + 1);
		}
		#pragma omp atomic
		pi += local_pi;
	}
	double endTime = omp_get_wtime();
	printf("pi: %lf, time taken: %lf seconds\n", pi, endTime - startTime);

	return 0;
}
