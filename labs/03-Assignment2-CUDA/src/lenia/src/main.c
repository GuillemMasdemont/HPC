#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <omp.h>
#include "lenia.h"

#define N 256
#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26
#define NUM_ORBIUMS 2

// Place two orbiums in the world with different angles. (y, x, angle)
// Orbiums size is 20x20, supproted angles are 0, 90, 180 and 270 degrees.
struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, N / 3, 0}, {N / 3, 0, 180}};

static int parse_thread_count(int argc, char **argv)
{
    if (argc < 2)
        return 0;

    errno = 0;
    char *end = NULL;
    long requested = strtol(argv[1], &end, 10);

    if (errno != 0 || end == argv[1] || *end != '\0' || requested <= 0 || requested > INT_MAX)
    {
        fprintf(stderr, "Usage: %s [thread_count]\n", argv[0]);
        fprintf(stderr, "thread_count must be a positive integer\n");
        return -1;
    }

    return (int)requested;
}

int main(int argc, char **argv)
{
    int thread_count = parse_thread_count(argc, argv);
    if (thread_count < 0)
        return EXIT_FAILURE;

    if (thread_count > 0)
        set_openmp_threads(thread_count);

    double start = omp_get_wtime();
    // Run the simulation
    double *world = evolve_lenia(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
    double stop = omp_get_wtime();
    printf("Execution time: %.3f\n", stop - start);
    free(world);
    return 0;
}