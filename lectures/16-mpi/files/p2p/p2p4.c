//
// send - receive - v4
//
// module load OpenMPI
// mpicc -o p2p4 p2p4.c
// salloc --reservation=fri --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
// mpirun --display-allocation --n 2 p2p4 

#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

#define buffsize 1000000

int main(int argc, char* argv[]) {
	int				taskid, ntasks;
	int				i;
	int				*sendbuff, *recvbuff;
	double			inittime, totaltime;
	MPI_Request		sendrequest, recvrequest;
	int				doneSend, doneRecv;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

	if(ntasks != 2){
		printf("Start exactly two processes!!!\n");
		MPI_Finalize();
		exit(1);
	}

	sendbuff = (int *)malloc(sizeof(int)*buffsize);
	recvbuff = (int *)malloc(sizeof(int)*buffsize);
	
	for(i=0; i<buffsize; i++)
		sendbuff[i] = i;

	MPI_Barrier(MPI_COMM_WORLD);
	inittime = MPI_Wtime();

	MPI_Isend(sendbuff, buffsize, MPI_INT, (taskid+1)%2, 0, 
			  MPI_COMM_WORLD, &sendrequest);
	MPI_Irecv(recvbuff, buffsize, MPI_INT, (taskid+1)%2, 
			  MPI_ANY_TAG, MPI_COMM_WORLD, &recvrequest);

	doneSend = 0;
	doneRecv = 0;
	while(!(doneSend && doneRecv)) {
		printf("%d", taskid);
		MPI_Test(&sendrequest, &doneSend, MPI_STATUS_IGNORE);
		MPI_Test(&recvrequest, &doneRecv, MPI_STATUS_IGNORE);
	}
	printf("\n");

	MPI_Barrier(MPI_COMM_WORLD);
	totaltime = MPI_Wtime() - inittime;

	if(taskid == 0)
		printf("Time: %f s\n", totaltime);

	free(recvbuff);
	free(sendbuff);

	MPI_Finalize();
}
