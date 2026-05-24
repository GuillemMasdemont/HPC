#ifndef LENIA_H
#define LENIA_H

#include <mpi.h>

struct orbium_coo {
    int row;
    int col;
    int angle;
};

/* Sequential implementation (lenia.c) */
double *evolve_lenia(const unsigned int rows, const unsigned int cols,
                     const unsigned int steps, const double dt,
                     const unsigned int kernel_size,
                     const struct orbium_coo *orbiums,
                     const unsigned int num_orbiums);

/* MPI parallel implementation — row-wise distribution (lenia_mpi.c) */
double *evolve_lenia_mpi(const unsigned int rows, const unsigned int cols,
                         const unsigned int steps, const double dt,
                         const unsigned int kernel_size,
                         const struct orbium_coo *orbiums,
                         const unsigned int num_orbiums,
                         int generate_gif);

#endif