// OpenMP variable scope - incorrect version
// gcc -fopenmp -o vars0 vars0.c
// srun --cpus-per-task=2 vars0

#include "omp.h"
#include <stdio.h>

int main(void) {
	int i, j;

	#pragma omp parallel for
	for (i =	0; i < 2; i++)
		for (j = 0; j < 10; j++)
			printf("%d-%d\n", i, j);

	return 0;
}
