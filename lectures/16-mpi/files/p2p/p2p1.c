//
// send - receive - v1
//
// module load OpenMPI
// mpicc -o p2p1 p2p1.c
// salloc --reservation=fri --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
// mpirun --display-allocation --n 2 p2p1 4000 // OK
// mpirun --display-allocation --n 2 p2p1 4001 // NOK

#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

int main(int argc, char* argv[]) {
	int				taskid, ntasks;
	int				i;
	int				*sendbuff, *recvbuff;
	double			inittime, totaltime;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

	if (ntasks != 2) {
		printf("Start exactly two processes!!!\n");
		MPI_Finalize();
		exit(1);
	}

	int buffsize = atoi(argv[1]); 

	sendbuff = (int *)malloc(sizeof(int)*buffsize);
	recvbuff = (int *)malloc(sizeof(int)*buffsize);

	for (i = 0; i<buffsize; i++)
		sendbuff[i] = i;

	MPI_Barrier(MPI_COMM_WORLD);
	inittime = MPI_Wtime();

	int size;
	MPI_Send(sendbuff, buffsize, MPI_INT, (taskid + 1) % 2,
		0, MPI_COMM_WORLD);
	MPI_Recv(recvbuff, buffsize, MPI_INT, (taskid + 1) % 2,
		MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	MPI_Barrier(MPI_COMM_WORLD);
	totaltime = MPI_Wtime() - inittime;

	if (taskid == 0)
		printf("Time: %f s, buffsize: %ld\n", totaltime, sizeof(int)*buffsize);

	free(recvbuff);
	free(sendbuff);

	MPI_Finalize();
}
