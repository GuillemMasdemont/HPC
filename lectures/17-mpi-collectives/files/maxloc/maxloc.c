// module load OpenMPI
// mpicc -o maxloc maxloc.c
// mpirun --display-allocation -np 4 ./maxloc

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

// custom datatype to support MPI_MAXLOC reduction

typedef struct { 
	double	value;
	int		id; 
} valueandid; 

int main(int argc, char* argv[]) {  
    valueandid	in, out; 
    int			myid, procs; 
 
	MPI_Init(&argc, &argv);				

    MPI_Comm_rank(MPI_COMM_WORLD, &myid); 
    MPI_Comm_size(MPI_COMM_WORLD, &procs); 

	srand( (unsigned)time( NULL ) * (myid+1) );

	in.value = rand()/(double)RAND_MAX; 
    in.id = myid; 
	
	printf("%d: %lf\n", in.id, in.value);
	fflush(stdout);

	MPI_Reduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD); 
	// ptr to local data, ptr to reduced data, data size, data type, operation, target process, communicator

	if (myid == 0) 
		printf("max = %lf, id = %d\n", out.value, out.id);

	MPI_Finalize();
}
