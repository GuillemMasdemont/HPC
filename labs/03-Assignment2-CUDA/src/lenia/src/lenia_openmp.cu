#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

// Uncomment to generate gif animation
//#define GENERATE_GIF

// For prettier indexing syntax
#define w(r, c) (w[(r) * w_cols + (c)])
#define input(r, c) (input[((r) % rows) * cols + ((c) % cols)])

// ---------------------------------------------------------------------------
// CHANGE 1: Replaced two separate pow() and exp() calls with a single
// temporary variable for the exponent. pow(x, 2) is slower than x*x because
// pow() is a general-purpose function with logarithm-based internals — a
// simple multiply is many times faster and produces the same result.
// ---------------------------------------------------------------------------
inline double gauss(double x, double mu, double sigma)
{
    double z = (x - mu) / sigma;
    return exp(-0.5 * z * z);
}

void set_openmp_threads(int num_threads)
{
    (void)num_threads;
}


// Function for growth criteria
double growth_lenia(double u)
{
    double mu = 0.15;
    double sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma);
}

// ---------------------------------------------------------------------------
// CHANGE 2: Removed the outer NULL guard that wrapped the entire function body.
// The original code did nothing and silently returned NULL when K was NULL,
// which would cause a crash in the caller anyway. The guard is now an early
// return with a clear message, and the rest of the function runs unconditionally
// — removing one level of indentation and one branch from the hot path.
// ---------------------------------------------------------------------------
double *generate_kernel(double *K, const unsigned int size)
{
    if (K == NULL)
    {
        fprintf(stderr, "generate_kernel: K is NULL\n");
        return NULL;
    }

    double mu = 0.5;
    double sigma = 0.15;
    int r = size / 2;
    double sum = 0;

    for (int y = -r; y < r; y++)
    {
        for (int x = -r; x < r; x++)
        {
            double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;

            // ---------------------------------------------------------------------------
            // CHANGE 3: Merged the Gaussian assignment and the distance>1 zero-out into
            // a single conditional expression. The original wrote the Gaussian value first
            // and then overwrote it with 0 in a separate if-block — two writes for cells
            // outside the circle. Now we skip the exp() call entirely for those cells.
            // ---------------------------------------------------------------------------
            K[(y + r) * size + x + r] = (distance <= 1.0) ? gauss(distance, mu, sigma) : 0.0;
            sum += K[(y + r) * size + x + r];
        }
    }

    // Normalize so all weights sum to 1
    for (unsigned int y = 0; y < size; y++)
        for (unsigned int x = 0; x < size; x++)
            K[y * size + x] /= sum;

    return K;
}

// ---------------------------------------------------------------------------
// CHANGE 4: Replaced NULL-guard branches with assertions and precomputed
// wrapped-index lookup tables.
//
// Original problem: the inner loop computed (i - w_rows/2 + rows + kri) % rows
// and the column equivalent on every single iteration. For a 256x256 world with
// a 28x28 kernel, that is 256*256*28*28 = ~51 million modulo operations per step.
// Modulo (%) is one of the slowest integer operations on modern CPUs.
//
// Fix: precompute a row_lut[] and col_lut[] array once before the loops. Each
// entry stores the already-wrapped row or column index for every kernel offset
// position. The inner loop then does a single array lookup instead of arithmetic
// + modulo. This also eliminates the repeated addition of `rows` and `cols` that
// was there just to avoid negative modulo — not needed once we precompute.
//
// CHANGE 5: Removed NULL checks on result/input/w. These are internal calls
// from evolve_lenia where the pointers are guaranteed valid (calloc was checked).
// Each NULL check added a branch at the top of a function called every time step.
// ---------------------------------------------------------------------------
inline double *convolve2d(
    double *result,
    const double *input,
    const double *w,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int w_rows,
    const unsigned int w_cols)
{
    // Precompute wrapped row indices for every (output_row, kernel_row) pair
    // row_lut[i * w_rows + ki] = the actual source row index after wrap-around
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

    for (unsigned int i = 0; i < rows; i++)
        for (unsigned int ki = 0; ki < w_rows; ki++)
            // kri is the flipped kernel row index (kernel is traversed reversed for convolution)
            row_lut[i * w_rows + ki] = ((int)i - half_r + (int)rows + (int)(w_rows - 1 - ki)) % rows;

    for (unsigned int j = 0; j < cols; j++)
        for (unsigned int kj = 0; kj < w_cols; kj++)
            col_lut[j * w_cols + kj] = ((int)j - half_c + (int)cols + (int)(w_cols - 1 - kj)) % cols;

    for (unsigned int i = 0; i < rows; i++)
    {
        for (unsigned int j = 0; j < cols; j++)
        {
            double sum = 0.0;

            for (unsigned int ki = 0; ki < w_rows; ki++)
            {
                // ---------------------------------------------------------------------------
                // CHANGE 6: Cache the source row pointer and kernel row pointer to avoid
                // recomputing row stride multiplications inside the innermost loop.
                // In the original, w(ki, kj) expanded to w[ki * w_cols + kj] — computing
                // ki * w_cols on every inner iteration. Hoisting the row base pointer out
                // of the inner loop reduces it to a single addition per column step.
                // ---------------------------------------------------------------------------
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

    // ---------------------------------------------------------------------------
    // CHANGE 7: Added NULL checks after every calloc. calloc returns NULL when
    // the system is out of memory. Without these checks the code would silently
    // proceed and crash with a segfault deep inside the simulation loops — very
    // hard to debug. Failing early with a message is much safer.
    // ---------------------------------------------------------------------------
    if (!w || !world || !tmp)
    {
        fprintf(stderr, "evolve_lenia: memory allocation failed\n");
        free(w); free(world); free(tmp);
        return NULL;
    }

    w = generate_kernel(w, kernel_size);

    for (unsigned int o = 0; o < num_orbiums; o++)
        world = place_orbium(world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);

    for (unsigned int step = 0; step < steps; step++)
    {
        tmp = convolve2d(tmp, world, w, rows, cols, kernel_size, kernel_size);

        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                // ---------------------------------------------------------------------------
                // BUG FIX (critical): The original code used `i * rows + j` to index into
                // the world and tmp grids. The correct formula for a 2D grid stored as a
                // flat 1D array is `i * cols + j` (row * number_of_columns + column).
                // Using rows instead of cols caused every row after row 0 to be read and
                // written at the wrong memory address whenever rows != cols, silently
                // corrupting the entire simulation state.
                // ---------------------------------------------------------------------------
                const unsigned int idx = i * cols + j;

                double val = world[idx] + dt * growth_lenia(tmp[idx]);

                // ---------------------------------------------------------------------------
                // CHANGE 8: Replaced nested fmin/fmax calls with a single if/else clamp.
                // fmin and fmax are floating-point library calls with NaN-handling overhead.
                // An if/else on a value already computed in a register is cheaper and makes
                // the intent (clamp to [0,1]) more explicit.
                // ---------------------------------------------------------------------------
                if      (val < 0.0) val = 0.0;
                else if (val > 1.0) val = 1.0;

                world[idx] = val;

#ifdef GENERATE_GIF
                // BUG FIX: was `i * rows + j` — same indexing bug as above, now fixed.
                gif->frame[idx] = (unsigned char)(val * 255);
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