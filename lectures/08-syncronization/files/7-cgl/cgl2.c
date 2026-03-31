// Conway's Game of Life
// gcc -fopenmp -o cgl2 cgl2.c
// srun --reservation=fri --cpus-per-task=16 cgl2

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

	#pragma omp parallel
	while (generations < MAXGENERATIONS) {
		#pragma omp for collapse(2)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				int neighs = count_neighbors(world, N, N, i, j);
				if (neighs == 3 || (world[i][j] == 1 && neighs == 2))
					worldNew[i][j] = 1;
				else
					worldNew[i][j] = 0;
			}

        #pragma omp barrier	// wait for all threads to finish before updating board_new

		#pragma omp master
        {
		    world_update(&world, &worldNew);
            generations++;
        }
		
        #pragma omp barrier	// wait for master thread to update board before next iteration
	}

	dt = omp_get_wtime()-dt;

	world_print(world, N, N);
	printf("Generation: %d\n", generations);

	world_free(world);
	world_free(worldNew);

	printf("Time: %lf\n", dt);

	return 0;
}
