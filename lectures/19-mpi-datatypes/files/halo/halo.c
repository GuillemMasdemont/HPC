#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define N 4

int main(int argc, char *argv[]) {
    int myid, procs;
    float *Mdata, **M;
    int i, j;
    MPI_Datatype rowtype, coltype;

    // Init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    if (procs != 2) {
        printf("Start two processes!\n");
        return 1;
    }

    // allocate matrices
    Mdata = (float *)calloc((N+2)*(N+2), sizeof(float));
    M = (float **)malloc((N+2)*sizeof(float *));
    for (i = 0; i < N+2; i++)
        M[i] = &Mdata[i*(N+2)];    

    // initialize matrices
    for (i = 1; i < N+1; i++)
        for (j = 1; j < N+1; j++) {
            M[i][j] = 10*i+j;
            if (myid == 1)
                M[i][j] = -M[i][j];
        }

    // print matrices
    printf("\nProc: %d:", myid);
    for (i = 0; i < N+2; i++) {
        printf("\n\t");
        for (j = 0; j < N+2; j++)
            printf("%3.0f ", M[i][j]);
    }
    printf("\n");
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);

    // create and commit datatypes
    MPI_Type_contiguous(N, MPI_FLOAT, &rowtype);        // array of N elements
    MPI_Type_vector(N, 1, N+2, MPI_FLOAT, &coltype);    // N blocks of 1 elements, stride N+2 between blocks
    MPI_Type_commit(&rowtype);
    MPI_Type_commit(&coltype);

    // exchange data
    MPI_Sendrecv(&M[1][1], 1, rowtype, (myid+1)%2, 0, 
                 &M[0][1], 1, rowtype, (myid+1)%2, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&M[N][1], 1, rowtype, (myid+1)%2, 0, 
                 &M[N+1][1], 1, rowtype, (myid+1)%2, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&M[1][1], 1, coltype, (myid+1)%2, 0, 
                 &M[1][0], 1, coltype, (myid+1)%2, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&M[1][N], 1, coltype, (myid+1)%2, 0, 
                 &M[1][N+1], 1, coltype, (myid+1)%2, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // print matrices
    MPI_Barrier(MPI_COMM_WORLD);
    printf("\nProc: %d:", myid);
    for (i = 0; i < N+2; i++) {
        printf("\n\t");
        for (j = 0; j < N+2; j++)
            printf("%3.0f ", M[i][j]);
    }
    printf("\n");
    fflush(stdout);

    MPI_Type_free(&rowtype);
    MPI_Type_free(&coltype);
    free(M);
    free(Mdata);

    MPI_Finalize();

    return 0;
}
