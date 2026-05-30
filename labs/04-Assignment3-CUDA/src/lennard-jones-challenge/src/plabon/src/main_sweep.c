#include <stdio.h>
#include "lennard-jones.h"

// Calls tune_sweep_and_print and writes the LJ_PRETUNE table to stdout.
// Tuner progress goes to stderr so the two streams can be separated:
//   ./lj_sweep.out > pretune_table.txt 2>sweep_progress.log
int main(void) {
    tune_sweep_and_print(
        1000,    // n_min
        150000,  // n_max
        1000,    // n_step
        0.95,    // density  (matches main.c)
        0.5,     // temperature (matches main.c)
        42       // seed
    );
    return 0;
}
