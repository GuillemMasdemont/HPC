# Parallel Pattern: Map

- replicates elemental function over every element of index set

- map replaces iterations of independent loops

- elemental functions should not modify global data that other instances (iterations) depend on

- serial and parallel execution of the map pattern (elements of index set (black) and elemental functions (blue))

  <img src="figures/map.png" alt="the map pattern", width="70%">

- examples:
  - gamma correction in images, color space conversions
  - Monte Carlo sampling, ray tracing

- map applies an elemental function to every element of a collection of data in parallel
  - elemental functions should have no side effects
  - no dependency among elements
  - can execute in any order

- embarrassingly parallel
  - one of the most efficient patterns
  - if you have many problems to solve, parallel solution can be as simple as running problems (serial code) on parallel execution nodes

- typically combined with other patterns
  - map does the basic computation, other patters follow
  - reduction, gather = serial random read + map, scan

- scalable implementation of map

  - a lot of care for best performance
  - threads
    - mandatory parallelism
    - separate thread for each element is not a good idea
  - tasks
    - optional parallelism
    - overhead and synchronization at the start and end when elemental functions vary in the amount of work

- the map pattern is a basis for vectorization and parallelization
  - map is related to SIMD, SPMD, SIMT
  - can be expressed as a sequence of vector operations

- parallel for construct in programming languages
  - map is parallelization of the serial iteration pattern where iterations are independent

- if dependencies and side-effects are avoided, map is deterministic

## Examples

### Monte Carlo $\pi$

- Integration
  - random shooting to the square

    <img src="figures/MonteCarloPi.png" alt="Monte Carlo Pi" width="70%">

  - statistics of $shots$ and $hits# - $shots$ inside the circle
  - ration $hits/shots$ is proportional to $\pi/4$
  - slow convergence, relative accuracy is $1/\square{shots}$
  - a basic unit of work that can be parallelized is one shot
    - a lot of overhead
    - better is to combine several shots to one task

### The Mandelbrot set

- an image of The Mandelbrot set from [Wikipedia](https://en.wikipedia.org/wiki/Mandelbrot_set)

    <img src="figures/Mandel_zoom_00_mandelbrot_set.jpg" alt="The Mandelbrot set" width="70%">

- the Mandelbrot set (black) is the set of all points $c$ in the complex plane that do not go to infinity with iterations
  - algorithm
    $$ z_0 = 0$$
    $$ z_{k-1} = z_{k}^2 + c $$
    $$ count(c) = \min_{0\leq k < K} (|z_k| \geq 2) $$
  - Divergence for large z
  - Compute the function up to some maximum value K

- serial control flow in elemental functions

- load imbalance

- implementation
  - opposed to vector operations like ´saxpy´,it cannot be efficiently computed on SIMD systems
  - best with SPMD or tiled SIMD

## Code Fusion and Cache fusion

- code fusion
  - map of sequence can avoid intermediate memory operations
  - intermediate results are written in registers
  - reduced memory bandwidth, cache and virtual memory problems
  - less synchronization at start/end of execution

    <img src="figures/fusion-code.png" alt="Code fusion" width="70%">

- cache fusion
  - maps broken to tiles
  - each tile is executed sequentially on one core
  - tile should fit in cache, avoid accessing main memory
  - loop with predefined size inside a parallel section of a code
  - each map has the same chunk size

    <img src="figures/fusion-cache.png" alt="Cache fusion" width="70%">

- code fusion is preferred

### Example: The Sieve of Erathostenes

- find primes in an interval $[1, n]$
  - find first prime ($k$)
  - repeat until $k \leq \sqrt{n}$
    - mark all multiplies of a prime
    - find next prime
  - count all primes
  - schematics of the algorithm

  <img src="figures/sieve-of-erathostenes.png" alt="The Sieve of Erathostenes" width="70%">

- demos
  - ```soe_basic```: one core
  - ```soe_chunk```: one core, sequentially chunk-by-chunk
  - ```soe_range```: multi core
  - ```soe_range_chunk```: multi core, sequentially chunk-by-chunk

## Patterns Related to Map

- stencil
  - access to neighbors to get inputs of elemental function
  - important is reading of data
    - data reuse
    - hardware specific for good results (cache size, GPU memory)
- workpile
  - work grows as it is consumed by map
  - could be implemented in OpenMP and OpenCL using explicit work queues
- divide-and-conquer
  - recursive division into smaller parallel subproblems until base is reached, which can be solved sequentially
  - combination of partition and map patterns
  - OpenMP: supported through tasking model
  - GPU programming: poor support for nested parallelism
