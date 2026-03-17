// Conway's Game of Life
// gcc -fopenmp -o cgl1 cgl1.c
// srun --reservation=fri --cpus-per-task=16 cgl1

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"
#include "board.h"

#define N 128
#define MAXITERS 100000

int main(void) {
	char** board, ** board_new;
	int iters = 0;

	srand(1573949136);
	board = board_initialize(N, N);
	board_new = board_initialize(N, N);

	board_print(board, N, N);

	double dt = omp_get_wtime();

	while (iters < MAXITERS) {
		#pragma omp parallel for collapse(2)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				int neighs = count_neighbors(board, N, N, i, j);
				if (neighs == 3 || (board[i][j] == 1 && neighs == 2))
					board_new[i][j] = 1;
				else
					board_new[i][j] = 0;
			}

		board_update(&board, &board_new);
		iters++;
	}

	dt = omp_get_wtime()-dt;

	board_print(board, N, N);
	printf("Iteration: %d\n", iters);

	board_free(board);
	board_free(board_new);

	printf("Time: %lf\n", dt);

	return 0;
}
