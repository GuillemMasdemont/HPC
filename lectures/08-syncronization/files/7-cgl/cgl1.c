// Conway's Game of Life
// gcc -fopenmp -o cgl1 cgl1.c
// srun --reservation=fri --cpus-per-task=16 cgl1

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"
#include "cgl.h"

#define N 128
#define MAXGENERATIONS 100000

int main(void) {
	char** world, ** worldNew;
	int generations = 0;

	srand(1573949136);
	world = world_initialize(N, N);
	worldNew = world_initialize(N, N);

	world_print(world, N, N);

	double dt = omp_get_wtime();

	while (generations < MAXGENERATIONS) {
		#pragma omp parallel for collapse(2)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				int neighs = count_neighbors(world, N, N, i, j);
				if (neighs == 3 || (world[i][j] == 1 && neighs == 2))
					worldNew[i][j] = 1;
				else
					worldNew[i][j] = 0;
			}

		world_update(&world, &worldNew);
		generations++;
	}

	dt = omp_get_wtime()-dt;

	world_print(world, N, N);
	printf("Generation: %d\n", generations);

	world_free(world);
	world_free(worldNew);

	printf("Time: %lf\n", dt);

	return 0;
}
