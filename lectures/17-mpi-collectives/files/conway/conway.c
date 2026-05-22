// module load OpenMPI
// mpicc -o conway conway.c
// mpirun --display-allocation -np 4 ./conway

#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "board.h"

#define N 20
#define MAXITERS 200

int main(int argc, char *argv[]) {
	int i, j, neighs;
	int iters = 0;

	char *boardptr = NULL;					// ptr to board
	char **board;							// board, 2D matrix, contiguous memory allocation!

	int procs, myid;			
	int mystart, myend, myrows;
	char **myboard;							// part of board that belongs to a process
	char **myboard_new;						// myboard_new is of the same size as myboard, needed for correct computation of iteration steps
	char *myrow_top, *myrow_bot;			// data (row) from top neighbour, data (row) from bottom neighbour

	MPI_Init(&argc, &argv);					// initialization
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);	// process ID
	MPI_Comm_size(MPI_COMM_WORLD, &procs);	// number of processes

	// initialize global board
	if (myid == 0) {
		srand(1573949136);
		board = board_initialize(N, N);
		boardptr = board[0];
		board_print(board, N, N);
	}
	// divide work
	mystart = N / procs * myid;				// determine scope of work for each process; process 0 also works on its own part
	myend = N / procs * (myid + 1);
	myrows = N / procs;

	// initialize my structures
	myboard = board_initialize(myrows, N);
	myboard_new = board_initialize(myrows, N);
	myrow_top = (char *)malloc(N * sizeof(char));
	myrow_bot = (char *)malloc(N * sizeof(char));

	// scatter initial matrix
	MPI_Scatter(boardptr, myrows * N, MPI_CHAR, 
				myboard[0], myrows * N, MPI_CHAR, 
				0, MPI_COMM_WORLD);
	// ptr to data (NULL on receiving processes), size of data sent to each process, data type, 
	// ptr to process data, size of received data, received data type, 
	// sender, communicator

	// do the calculation
	while (iters < MAXITERS) {
		// exchange borders with neighboring processes
		MPI_Sendrecv(myboard[0], N, MPI_CHAR, (myid + procs - 1) % procs, 0,
					 myrow_bot, N, MPI_CHAR, (myid + 1) % procs, 0,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		// ptr to send data, send data size, send data type, receiver, message tag,
		// ptr to received data, received data size, received data type, sender, message tag,
		// communicator, status
		MPI_Sendrecv(myboard[myrows - 1], N, MPI_CHAR, (myid + 1) % procs, 1,
					 myrow_top, N, MPI_CHAR, (myid + procs - 1) % procs, 1, 
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		// do the computation of my part
		for (i = 0; i < myrows; i++)
			for (j = 0; j < N; j++) {
				neighs = count_neighbours_mpi(myboard, myrow_top, myrow_bot, myrows, N, i, j);
				if (neighs == 3 || (myboard[i][j] == 1 && neighs == 2))
					myboard_new[i][j] = 1;
				else
					myboard_new[i][j] = 0;
			}
		iters++;
		// swap boards (iter --> iter + 1)
		board_update(&myboard, &myboard_new);
	}
	
	// gather results
	MPI_Gather(myboard[0], myrows * N, MPI_CHAR, 
			   boardptr, myrows * N, MPI_CHAR, 
			   0, MPI_COMM_WORLD);
	// data to send, send data size, data type,
	// gathered data, received data size, data type,
	// gathering process, communicator

	// display
	if (myid == 0)
		board_print(board, N, N);

	// free memory
	if (myid == 0)
		board_free(board);
	board_free(myboard);
	free(myrow_top);
	free(myrow_bot);

	MPI_Finalize();			// finalize MPI

	return 0;
}
