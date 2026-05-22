#include <stdio.h>
#include <stdlib.h>
#include "lenia.h"

#define N 128
#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26
#define NUM_ORBIUMS 2


struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, N / 3, 0}, {N / 3, 0, 180}};

int main(int argc, char* argv[])
{   
    int		    myid, procs;
    char node_name[MPI_MAX_PROCESSOR_NAME]; 
	int name_len;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);	// process ID
	MPI_Comm_size(MPI_COMM_WORLD, &procs);	// number of processes involved in communication
    MPI_Get_processor_name( node_name, &name_len ); // compute node name
    printf("Hello from process %d of %d in node %s\n", myid, procs, node_name);
    
    double start = MPI_Wtime();
    double *world = evolve_lenia(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
    double stop = MPI_Wtime();
    printf("Execution time: %.3f\n", stop - start);
    free(world);
    MPI_Finalize();
    return 0;
}