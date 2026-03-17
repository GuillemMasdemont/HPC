// OpenMP variable scope - definition in local scope
// gcc -fopenmp -o vars2 vars2.c
// srun --cpus-per-task=2 vars2

#include "omp.h"
#include <stdio.h>

int main(void) {
	#pragma omp parallel for
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 10; j++)
			printf("%d-%d\n", i, j);

	return 0;
}
