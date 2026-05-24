#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"


// Uncomment to generate gif animation
//#define GENERATE_GIF

// For prettier indexing syntax
#define w(r, c) (w[(r) * w_cols + (c)])
#define input(r, c) (input[((r) % rows) * cols + ((c) % cols)])

// Function to calculate Gaussian
inline double gauss(double x, double mu, double sigma)
{
    return exp(-0.5 * pow((x - mu) / sigma, 2));
}

// Function for growth criteria
double growth_lenia(double u)
{
    double mu = 0.15;
    double sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma); // Baseline -1, peak +1
}

// Function to generate convolution kernel
double *generate_kernel(double *K, const unsigned int size)
{
    // Construct ring convolution filter
    double mu = 0.5;
    double sigma = 0.15;
    int r = size / 2;
    double sum = 0;
    if (K != NULL)
    {
        for (int y = -r; y < r; y++)
        {
            for (int x = -r; x < r; x++)
            {
                double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;
                K[(y + r) * size + x + r] = gauss(distance, mu, sigma);
                if (distance > 1)
                {
                    K[(y + r) * size + x + r] = 0; // Cut at d=1
                }
                sum += K[(y + r) * size + x + r];
            }
        }
        // Normalize
        for (unsigned int y = 0; y < size; y++)
        {
            for (unsigned int x = 0; x < size; x++)
            {
                K[y * size + x] /= sum;
            }
        }
    }
    return K;
}

// Function to perform convolution on input using kernel w
inline double *convolve2d(double *result, const double *input, const double *w, const unsigned int rows, const unsigned int cols, const unsigned int w_rows, const unsigned int w_cols)
{
    if (result != NULL && input != NULL && w != NULL)
    {
        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                double sum = 0;
                for (int ki = w_rows - 1, kri = 0; ki >= 0; ki--, kri++)
                {
                    for (int kj = w_cols - 1, kcj = 0; kj >= 0; kj--, kcj++)
                    {
                        sum += w(ki, kj) * input((i - w_rows / 2 + rows + kri), (j - w_cols / 2 + cols + kcj));
                    }
                }
                result[i * cols + j] = sum;
            }
        }
    }
    return result;
}

// Function to evolve Lenia
double *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums)
{

#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif(
        "lenia.gif",     /* file name */
        cols, rows,      /* canvas size */
        inferno_pallete, /*pallete*/
        8,               /* palette depth == log2(# of colors) */
        -1,              /* no transparency */
        0                /* infinite loop */
    );
#endif

    // Allocate memory
    double *w = (double *)calloc(kernel_size * kernel_size, sizeof(double));
    double *world = (double *)calloc(rows * cols, sizeof(double));
    double *tmp = (double *)calloc(rows * cols, sizeof(double));

    // Generate convolution kernel
    w=generate_kernel(w,kernel_size);

    // Place orbiums
    for (unsigned int o = 0; o < num_orbiums; o++)
    {
        world = place_orbium(world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    // Lenia Simulation
    for (unsigned int step = 0; step < steps; step++)
    {
        // Convolution
        tmp = convolve2d(tmp, world, w, rows, cols, kernel_size, kernel_size);
        
        // Evolution
        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                world[i * rows + j] += dt * growth_lenia(tmp[i * rows + j]);
                world[i * rows + j] = fmin(1, fmax(0, world[i * rows + j])); // Clip between 0 and 1
#ifdef GENERATE_GIF
                gif->frame[i * rows + j] = world[i * rows + j] * 255;
#endif
            }
        }
#ifdef GENERATE_GIF
        ge_add_frame(gif, 5);
#endif
    }
#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif
    free(w);
    free(tmp);
    return world;
}

// =============================================================================
// MPI PARALLEL IMPLEMENTATION — row-wise work distribution
// =============================================================================

/* --------------------------------------------------------------------------
 * Local convolution on a halo-padded buffer
 *
 * Buffer layout:
 *   [ ghost_top: halo rows ][ real: local_rows rows ][ ghost_bot: halo rows ]
 *
 * result  – local_rows × cols (no halo)
 * buf     – (local_rows + 2*halo) × cols
 *
 * For real row i the kernel sweeps buf rows [i, i + kernel_size), which
 * is always in [0, local_rows + 2*halo) because:
 *   min: i=0,             kri=0            → buf row 0
 *   max: i=local_rows-1,  kri=kernel_size-1 → buf row local_rows + 2*halo - 2
 *
 * Column wrapping is done with modulo.
 * -------------------------------------------------------------------------- */
static void convolve_local(double *result, const double *buf,
                           int local_rows, int cols,
                           const double *kernel, int kernel_size)
{
    int halo = kernel_size / 2;
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++) {
                for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++) {
                    /* buf row index: halo offset cancels with -kernel_size/2 */
                    int row_idx = i + kri;
                    int col_idx = (j + kcj - halo + cols) % cols;
                    sum += kernel[ki * kernel_size + kj] *
                           buf[(size_t)row_idx * cols + col_idx];
                }
            }
            result[(size_t)i * cols + j] = sum;
        }
    }
}

/* --------------------------------------------------------------------------
 * Halo exchange (ring topology, periodic / wrap-around boundaries)
 *
 * Normal case (local_rows >= halo):
 *   Two MPI_Sendrecv calls exchange halo-width strips with neighbours.
 *
 * Edge case (local_rows < halo):
 *   Occurs for 128×128 grid with ≥16 processes (4 rows/proc < 13 = halo).
 *   Fall back to MPI_Allgatherv: gather the full grid to every process,
 *   then copy the required rows into the ghost areas using global modular
 *   indexing.  This is only used for very small grids, so cost is low.
 * -------------------------------------------------------------------------- */
static void exchange_halos(double *local_buf, int local_rows, int cols,
                           int halo, int *sendcounts, int *displs,
                           int my_start, int total_rows, int myid, int procs)
{
    if (local_rows >= halo) {
        int prev = (myid - 1 + procs) % procs;
        int next = (myid + 1) % procs;

        /* bottom real rows → next; top ghost ← prev */
        MPI_Sendrecv(
            local_buf + (size_t)local_rows * cols,
            halo * cols, MPI_DOUBLE, next, 0,
            local_buf,
            halo * cols, MPI_DOUBLE, prev, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* top real rows → prev; bottom ghost ← next */
        MPI_Sendrecv(
            local_buf + (size_t)halo * cols,
            halo * cols, MPI_DOUBLE, prev, 1,
            local_buf + (size_t)(halo + local_rows) * cols,
            halo * cols, MPI_DOUBLE, next, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } else {
        double *full = malloc((size_t)total_rows * cols * sizeof(double));

        MPI_Allgatherv(local_buf + (size_t)halo * cols,
                       local_rows * cols, MPI_DOUBLE,
                       full, sendcounts, displs, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        for (int gi = 0; gi < halo; gi++) {
            int grow = (my_start - halo + gi + total_rows) % total_rows;
            memcpy(local_buf + (size_t)gi * cols,
                   full    + (size_t)grow * cols,
                   (size_t)cols * sizeof(double));
        }
        for (int gi = 0; gi < halo; gi++) {
            int grow = (my_start + local_rows + gi) % total_rows;
            memcpy(local_buf + (size_t)(halo + local_rows + gi) * cols,
                   full    + (size_t)grow * cols,
                   (size_t)cols * sizeof(double));
        }
        free(full);
    }
}

/* --------------------------------------------------------------------------
 * MPI-parallel Lenia evolution — row-wise work distribution
 *
 * Each process owns a contiguous strip of rows.  Before each convolution
 * step, ghost rows (halo = kernel_size/2 rows from each neighbour) are
 * exchanged so the convolution reads correct wrap-around values.
 *
 * generate_gif: when non-zero, gather every frame to rank 0 and write
 *               lenia.gif using the gifenc library.
 *
 * Returns: pointer to the final rows × cols world on rank 0, NULL elsewhere.
 *          Caller on rank 0 must free() the pointer.
 * -------------------------------------------------------------------------- */
double *evolve_lenia_mpi(const unsigned int rows, const unsigned int cols,
                         const unsigned int steps, const double dt,
                         const unsigned int kernel_size,
                         const struct orbium_coo *orbiums,
                         const unsigned int num_orbiums,
                         int generate_gif)
{
    int myid, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    int halo = (int)kernel_size / 2;

    /* --- Row distribution (handles rows not evenly divisible by procs) --- */
    int *sendcounts = malloc(procs * sizeof(int));
    int *displs     = malloc(procs * sizeof(int));
    int rem = (int)rows % procs;
    displs[0] = 0;
    for (int i = 0; i < procs; i++) {
        sendcounts[i] = ((int)rows / procs + (i < rem ? 1 : 0)) * (int)cols;
        if (i > 0) displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
    int local_rows = sendcounts[myid] / (int)cols;
    int my_start   = displs[myid]     / (int)cols;

    /* --- Local buffer: [ghost_top | real rows | ghost_bot] --------------- */
    int buf_rows = local_rows + 2 * halo;
    double *local_buf = calloc((size_t)buf_rows * cols, sizeof(double));
    double *local_tmp = malloc((size_t)local_rows * cols * sizeof(double));
    double *kernel_w  = calloc(kernel_size * kernel_size, sizeof(double));
    generate_kernel(kernel_w, kernel_size);

    /* --- Root: initialise full world and scatter rows to all processes --- */
    double *world = NULL;
    if (myid == 0) {
        world = calloc((size_t)rows * cols, sizeof(double));
        for (unsigned int o = 0; o < num_orbiums; o++)
            place_orbium(world, rows, cols,
                         (unsigned int)orbiums[o].row,
                         (unsigned int)orbiums[o].col,
                         (unsigned int)orbiums[o].angle);
    }

    MPI_Scatterv(world,
                 sendcounts, displs, MPI_DOUBLE,
                 local_buf + (size_t)halo * cols, local_rows * cols, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* --- Optional GIF (root only) --------------------------------------- */
    ge_GIF *gif = NULL;
    if (generate_gif && myid == 0) {
        gif = ge_new_gif("lenia.gif",
                         (uint16_t)cols, (uint16_t)rows,
                         inferno_pallete, 8, -1, 0);
    }

    /* --- Main simulation loop ------------------------------------------- */
    for (unsigned int step = 0; step < steps; step++) {

        /* 1. Exchange halo rows with neighbours */
        exchange_halos(local_buf, local_rows, (int)cols, halo,
                       sendcounts, displs, my_start, (int)rows,
                       myid, procs);

        /* 2. Convolve (reads ghost + real rows, writes only real rows) */
        convolve_local(local_tmp, local_buf, local_rows, (int)cols,
                       kernel_w, (int)kernel_size);

        /* 3. Apply growth function and clip to [0, 1] */
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < (int)cols; j++) {
                double *cell = &local_buf[(size_t)(halo + i) * cols + j];
                *cell += dt * growth_lenia(local_tmp[(size_t)i * cols + j]);
                *cell = fmin(1.0, fmax(0.0, *cell));
            }
        }

        /* 4. Optionally gather to root and write GIF frame */
        if (generate_gif) {
            MPI_Gatherv(local_buf + (size_t)halo * cols, local_rows * cols, MPI_DOUBLE,
                        world, sendcounts, displs, MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            if (myid == 0 && gif != NULL) {
                for (unsigned int px = 0; px < rows * cols; px++)
                    gif->frame[px] = (uint8_t)(world[px] * 255);
                ge_add_frame(gif, 5);
            }
        }
    }

    /* --- Final gather to rank 0 ----------------------------------------- */
    /* When GIF mode is active, the last loop iteration already gathered. */
    if (!generate_gif) {
        MPI_Gatherv(local_buf + (size_t)halo * cols, local_rows * cols, MPI_DOUBLE,
                    world, sendcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (gif != NULL)
        ge_close_gif(gif);

    free(local_buf);
    free(local_tmp);
    free(kernel_w);
    free(sendcounts);
    free(displs);

    return world; /* non-NULL only on rank 0 */
}
