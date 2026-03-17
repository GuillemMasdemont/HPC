// OpenMP variable scope - private clause
// gcc -fopenmp -o vars1 vars1.c
// srun --cpus-per-task=2 vars1

#include "omp.h"
#include <stdio.h>

int main(void) {
	int i, j;

	#pragma omp parallel for private(j)
	for (i = 0; i < 2; i++)
		for (j = 0; j < 10; j++)
			printf("%d-%d\n", i, j);

	return 0;
}
