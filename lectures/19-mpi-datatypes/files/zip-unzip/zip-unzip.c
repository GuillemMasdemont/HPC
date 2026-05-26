#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define NUM_ELEMENTS 5

int main(int argc, char *argv[]) {
    int myid, procs;
    int i, j;
    char local_array[NUM_ELEMENTS];
    char *collected_array = NULL;
    MPI_Datatype datatype_base, datatype_zip;

    // Init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    // create new receiving datatype
    // we will transfer only one element of new datatype from each proc
    MPI_Type_vector(NUM_ELEMENTS, 1, procs, MPI_CHAR, &datatype_base);
    // blocks, elements per block, stride between blocks, old datatype, new datatype
    MPI_Type_commit(&datatype_base);
    // change the extend to (dirty) trick gather to do proper interleaving, size does not change remains
    // extend is size of bytes, which datatype uses in memory (unused bytes are not counted)
    MPI_Type_create_resized(datatype_base, 0, 1, &datatype_zip);
    // existing type, new lower bound, new upper bound, new data type
    MPI_Type_commit(&datatype_zip);

    if (myid == 0) {
        MPI_Aint lb, extent;
        int s;
        MPI_Type_get_extent(datatype_base, &lb, &extent);
        MPI_Type_size(datatype_base, &s);
        printf("datatype_base: lower bound: %d upper bound: %d size: %ld\n", lb, lb+extent, s);
        MPI_Type_get_extent(datatype_zip, &lb, &extent);
        MPI_Type_size(datatype_zip, &s);
        printf("datatype_zip:  lower bound: %d upper bound: %d size: %ld\n\n", lb, lb+extent, s);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // prepare data to zip
    // 0: A A A ...
    // 1: B B B ...
    // 2: C C C ...
    for (i = 0; i < NUM_ELEMENTS; i++)
        local_array[i] = 'A' + myid;

    // print unzipped data
    printf("Proc %d: ", myid);
    for (i = 0; i < NUM_ELEMENTS; i++)
        printf("%c", local_array[i]);
    printf("\n");

    if(myid == 0)
        collected_array = (char *)malloc(procs*NUM_ELEMENTS*sizeof(char));
        
    MPI_Gather(local_array, NUM_ELEMENTS, MPI_CHAR, collected_array, 1, datatype_zip, 0, MPI_COMM_WORLD);
    // each process sends NUM_ELEMENTS of type MPI_CHAR
    // root process collects 1 datatype_zip from each process

    // Print zipped data
    if (myid == 0) {
        printf("Result: ");
        for (i = 0; i < procs * NUM_ELEMENTS; i++) 
            printf("%c", collected_array[i]);
        printf("\n\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // prepare data to unzip
    // 0 1 2 ... 0 1 2 ... 0 1 2 
    if (myid == 0)
        for (i = 0; i < procs * NUM_ELEMENTS; i++) 
            collected_array[i] = '0' + i % procs;

    // print zipped data
    if (myid == 0) {
        printf("Data:   ");
        for (i = 0; i < procs * NUM_ELEMENTS; i++) 
            printf("%c", collected_array[i]);
        printf("\n");
    }

    MPI_Scatter(collected_array, 1, datatype_zip, local_array, NUM_ELEMENTS, MPI_CHAR, 0, MPI_COMM_WORLD);

    // print unzipped data
    printf("Proc %d: ", myid);
    for (i = 0; i < NUM_ELEMENTS; i++)
        printf("%c", local_array[i]);
    printf("\n");

    if (myid == 0)
        free(collected_array);

    MPI_Type_free(&datatype_base);
    MPI_Type_free(&datatype_zip);

    MPI_Finalize();

    return 0;
}
