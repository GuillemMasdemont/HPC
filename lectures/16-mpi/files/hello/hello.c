// module load OpenMPI
// mpicc -o hello hello.c
// run through sbatch script hello.sh

#include <stdio.h>
#include <string.h>
#include "mpi.h"

#define BUF_SIZE 80

int main(int argc, char* argv[])
{
	int		    myid, procs;
	char	    buffer[BUF_SIZE];
	int		    i, provided;
    MPI_Status  status;

	// MPI_Init(&argc, &argv);											// MPI-1 initialization
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);	
	// MPI-2 initialization adds option for thread-safe operation 
	// MPI_THREAD_SINGLE is the same as MPI-1 initialization
	// use MPI_THREAD_MULTIPLE when MPI is combined with OpenMP

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);	// process ID
	MPI_Comm_size(MPI_COMM_WORLD, &procs);	// number of processes involved in communication

	// get node name
    char nodename[MPI_MAX_PROCESSOR_NAME];
    int nodename_len;
    MPI_Get_processor_name(nodename, &nodename_len);

	if (myid == 0)  // master collects and prints messages from other processes
	{		
		printf("Hello from master process %d/%d running on %s!\n", myid, procs, nodename);
		for (i = 1; i < procs; i++)
		{
			MPI_Recv(buffer, BUF_SIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			// ptr to data, data size, data type, source id, message tag, communicator, status (MPI_STATUS_IGNORE)
			// MPI_ANY_SOURCE allows to receive messages in any order, we can enforce order by replacing it with i
			printf("%s", buffer);
		}
	}
	else    // other processes generate messages and send them to master
	{
		sprintf(buffer, "Hello from process %d/%d running on %s!\n", myid, procs, nodename);
		MPI_Send(buffer, strlen(buffer)+1, MPI_CHAR, 0, myid, MPI_COMM_WORLD);
		// ptr to data, data size, data type, destination process, message tag, communicator
	}

	MPI_Finalize();

	return 0;
}
