#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

struct vehicle {
    int mass;
    float position;
    float velocity;
    float acceleration;
};

int main(int argc, char *argv[]) {
    int         myid, procs;
   	MPI_Aint	lowerboundint, extentint;
    struct      vehicle *vehicles_collected = NULL;
    struct      vehicle vehicle_local;

    // Init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    // create typevehicle composed of 1 x int and 3 x float
    // prepare structures
    MPI_Type_get_extent(MPI_INT, &lowerboundint, &extentint);	// what is the length of MPI_INT?
	MPI_Datatype	inputtype[2] = {MPI_INT, MPI_FLOAT};                
	int				blocks[2] = {1, 3};
	MPI_Aint		displacement[2] = {0, extentint};           // start position of blocks in typevehicle
    // create and commit new type
	MPI_Datatype	typevehicle;    
	MPI_Type_create_struct(2, blocks, displacement, inputtype, &typevehicle);
	MPI_Type_commit(&typevehicle);

    // initialize local strucutre
    vehicle_local.mass = 1000*(myid+1);
    vehicle_local.position = 100*(myid+1);
    vehicle_local.velocity = 10*(myid+1);
    vehicle_local.acceleration = 1*(myid+1);

    // initialize array of vehicles on root
    if (myid == 0)
        vehicles_collected = (struct vehicle *)malloc(procs * sizeof(struct vehicle));

    // gather
    MPI_Gather(&vehicle_local, 1, typevehicle, vehicles_collected, 1, typevehicle, 0, MPI_COMM_WORLD);

    // display
    if (myid == 0)
        for (int i = 0; i < procs; i++) {
            printf("Vehicle %d:\n", i);
            printf("\tmass: %d\n", vehicles_collected[i].mass);
            printf("\tposition: %.0f\n", vehicles_collected[i].position);
            printf("\tvelocity: %.0f\n", vehicles_collected[i].velocity);
            printf("\tacceleration: %.0f\n", vehicles_collected[i].acceleration);
        }

    // free structures and finalize
	MPI_Type_free(&typevehicle);
    free(vehicles_collected);

    MPI_Finalize();

    return 0;
}
