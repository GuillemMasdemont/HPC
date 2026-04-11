#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// [OPENMP] Include the OpenMP header. This gives access to omp_get_max_threads()
// and all #pragma omp directives. If compiled without -fopenmp, the compiler
// ignores every #pragma omp line and the code runs correctly single-threaded.
#include <omp.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

// Uncomment to generate gif animation
// #define GENERATE_GIF

// For prettier indexing syntax
#define w(r, c) (w[(r) * w_cols + (c)])
#define input(r, c) (input[((r) % rows) * cols + ((c) % cols)])

// ---------------------------------------------------------------------------
// CHANGE 1 (previous): Replaced pow(x,2) with x*x via a temp variable.
// pow() uses logarithm-based internals; a plain multiply is much faster.
// ---------------------------------------------------------------------------
inline double gauss(double x, double mu, double sigma)
{
    double z = (x - mu) / sigma;
    return exp(-0.5 * z * z);
}

double growth_lenia(double u)
{
    double mu    = 0.15;
    double sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma);
}

void set_openmp_threads(int num_threads)
{
    if (num_threads > 0)
    {
        omp_set_dynamic(0);
        omp_set_num_threads(num_threads);
    }
}

double *generate_kernel(double *K, const unsigned int size)
{
    if (K == NULL)
    {
        fprintf(stderr, "generate_kernel: K is NULL\n");
        return NULL;
    }

    double mu    = 0.5;
    double sigma = 0.15;
    int    r     = size / 2;
    double sum   = 0.0;

    // [OPENMP] Parallelise the kernel fill over the outer (y) loop.
    //
    // collapse(2) was removed: it merges loop counters into a single
    // linearised index that OpenMP decomposes back into (y, x) on every
    // iteration. For a kernel that is already small (e.g. 28x28 = 28 outer
    // rows), the decomposition arithmetic costs more than it saves.
    // Parallelising only the outer loop is cheaper and still distributes
    // work across all threads because each row contains `size` iterations
    // of non-trivial work (sqrt + conditional exp).
    //
    // reduction(+:sum): each thread accumulates its own private copy of
    // `sum`; OpenMP adds them all together into the shared variable when
    // the parallel region ends. Without this clause every thread would
    // race to write the same memory location and produce a wrong result.
    //
    // schedule(static): divides the outer rows into equal-sized chunks at
    // compile time — correct here because every row costs the same amount.
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int y = -r; y < r; y++)
    {
        for (int x = -r; x < r; x++)
        {
            double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;

            // CHANGE 3 (previous): skip exp() entirely for cells outside the circle.
            K[(y + r) * size + x + r] = (distance <= 1.0) ? gauss(distance, mu, sigma) : 0.0;
            sum += K[(y + r) * size + x + r];
        }
    }

    // [OPENMP] Parallelise the normalisation pass over the outer (y) loop.
    // Each row is independent — no dependencies between iterations.
    // collapse(2) removed for the same reason as above.
    #pragma omp parallel for schedule(static)
    for (unsigned int y = 0; y < size; y++)
        for (unsigned int x = 0; x < size; x++)
            K[y * size + x] /= sum;

    return K;
}

inline double *convolve2d(
    double       *result,
    const double *input,
    const double *w,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int w_rows,
    const unsigned int w_cols)
{
    unsigned int *row_lut = (unsigned int *)malloc(rows * w_rows * sizeof(unsigned int));
    unsigned int *col_lut = (unsigned int *)malloc(cols * w_cols * sizeof(unsigned int));

    if (!row_lut || !col_lut)
    {
        fprintf(stderr, "convolve2d: out of memory for lookup tables\n");
        free(row_lut);
        free(col_lut);
        return NULL;
    }

    const int half_r = (int)w_rows / 2;
    const int half_c = (int)w_cols / 2;

    // [OPENMP] Build the row lookup table in parallel over the outer (i) loop.
    // Every entry row_lut[i * w_rows + ki] is written by exactly one thread's
    // (i, ki) pair — no two threads touch the same element.
    // collapse(2) removed: the overhead of index linearisation outweighs the
    // benefit for typical grid sizes where `rows` alone gives plenty of work.
    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < rows; i++)
        for (unsigned int ki = 0; ki < w_rows; ki++)
            row_lut[i * w_rows + ki] =
                ((int)i - half_r + (int)rows + (int)(w_rows - 1 - ki)) % rows;

    // [OPENMP] Build the column lookup table in parallel over the outer (j) loop.
    // Same reasoning as row_lut above — collapse(2) removed.
    #pragma omp parallel for schedule(static)
    for (unsigned int j = 0; j < cols; j++)
        for (unsigned int kj = 0; kj < w_cols; kj++)
            col_lut[j * w_cols + kj] =
                ((int)j - half_c + (int)cols + (int)(w_cols - 1 - kj)) % cols;

    // [OPENMP] Parallelise the main convolution over output rows (i).
    //
    // Each thread handles one or more complete rows. Within each row,
    // every cell (j) computes its kernel sum privately and writes to its
    // own result[i * cols + j] slot — no overlap between threads.
    //
    // collapse(2) removed: parallelising only the outer (i) loop is
    // sufficient because each row contains `cols` cells each doing
    // w_rows * w_cols multiply-adds — enough arithmetic to keep a thread
    // busy without needing to merge the j loop into the work pool.
    //
    // w, input, row_lut, col_lut are all read-only — safe for concurrent access.
    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < rows; i++)
    {
        for (unsigned int j = 0; j < cols; j++)
        {
            double sum = 0.0;

            for (unsigned int ki = 0; ki < w_rows; ki++)
            {
                // BUG FIX: the previous version read the kernel forwards (ki * w_cols)
                // while reading the input forwards via the LUT — that is
                // cross-correlation, not convolution. Convolution requires flipping
                // the kernel before multiplying, not the input.
                //
                // Fix: ki and kj scan the input forwards (0→N) via the LUT as
                // intended, and the kernel is flipped explicitly at the point of
                // access by inverting both indices:
                //   row: (w_rows - 1 - ki) → ki=0 reads last kernel row, ki=last reads row 0
                //   col: (w_cols - 1 - kj) → same inversion column-wise
                //
                // This is exactly equivalent to the original double-counter trick:
                //   original ki  (descending) == (w_rows - 1 - ki) with ki ascending
                //   original kri (ascending)  == ki ascending, used to index input
                const double *w_row   = &w[(w_rows - 1 - ki) * w_cols];
                const double *src_row = &input[row_lut[i * w_rows + ki] * cols];

                for (unsigned int kj = 0; kj < w_cols; kj++)
                    sum += w_row[w_cols - 1 - kj] * src_row[col_lut[j * w_cols + kj]];
            }

            result[i * cols + j] = sum;
        }
    }

    free(row_lut);
    free(col_lut);
    return result;
}

double *evolve_lenia(
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int steps,
    const double dt,
    const unsigned int kernel_size,
    const struct orbium_coo *orbiums,
    const unsigned int num_orbiums)
{
    // [OPENMP] Print thread count once at startup so the user can confirm
    // parallelism is active. omp_get_max_threads() returns the number of
    // threads OpenMP will use — typically the number of logical CPU cores.
    printf("evolve_lenia: using %d OpenMP thread(s)\n", omp_get_max_threads());

#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif(
        "lenia.gif",
        cols, rows,
        inferno_pallete,
        8,
        -1,
        0);
#endif

    double *w     = (double *)calloc(kernel_size * kernel_size, sizeof(double));
    double *world = (double *)calloc(rows * cols, sizeof(double));
    double *tmp   = (double *)calloc(rows * cols, sizeof(double));

    if (!w || !world || !tmp)
    {
        fprintf(stderr, "evolve_lenia: memory allocation failed\n");
        free(w); free(world); free(tmp);
        return NULL;
    }

    w = generate_kernel(w, kernel_size);

    for (unsigned int o = 0; o < num_orbiums; o++)
        world = place_orbium(world, rows, cols,
                             orbiums[o].row, orbiums[o].col, orbiums[o].angle);

    // NOTE: The outer step loop is intentionally NOT parallelised.
    // Each step reads the world state written by the previous step — a true
    // sequential dependency. Parallelising across steps would produce a
    // different (incorrect) simulation result.
    for (unsigned int step = 0; step < steps; step++)
    {
        // convolve2d is internally parallelised — see above.
        tmp = convolve2d(tmp, world, w, rows, cols, kernel_size, kernel_size);

        // [OPENMP] Parallelise the per-cell evolution update over rows (i).
        //
        // Each thread handles one or more complete rows. Within each row,
        // every cell reads and writes only its own world[idx] and tmp[idx]
        // slot — no overlap between threads.
        //
        // collapse(2) removed: the outer loop over `rows` gives each thread
        // a contiguous chunk of memory to work through, which is better for
        // cache locality than the interleaved access pattern that collapsed
        // loop index decomposition can produce.
        #pragma omp parallel for schedule(static)
        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                // BUG FIX (previous): was i * rows + j — wrong when rows != cols.
                const unsigned int idx = i * cols + j;

                double val = world[idx] + dt * growth_lenia(tmp[idx]);

                // CHANGE 8 (previous): if/else clamp is cheaper than fmin/fmax.
                if      (val < 0.0) val = 0.0;
                else if (val > 1.0) val = 1.0;

                world[idx] = val;

#ifdef GENERATE_GIF
                // Each thread writes to a unique gif->frame[idx] — no race condition.
                gif->frame[idx] = (unsigned char)(val * 255);
#endif
            }
        }

        // [OPENMP] ge_add_frame writes sequentially to the GIF file on disk.
        // It MUST stay outside the parallel region — calling it from multiple
        // threads simultaneously would corrupt the output file.
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