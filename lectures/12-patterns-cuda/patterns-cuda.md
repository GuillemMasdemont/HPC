# Patterns with CUDA Programming

## Stencil

- special case of map
  - 1D or multiple dimensions

  <img src="figures/stencil-1D.png" alt="1D stencil" width="50%" />

- has regular data access pattern
  - each output depends on a neighborhood of inputs (black)
  - inputs have fixed offsets relative to the output
  - can be implemented as
    - set of random reads for each output
    - shifts
- applications
  - image and signal processing (convolution)
  - physics, mechanical engineering, CFD (PDE solvers over regular grids)
  - cellular automata
- different neighborhoods
  - square compact, ..., sparse
  - cache optimizations
  - stencils reuse samples required for neighboring elements
- boundaries of grids given to a processor (red dashed lines)
  - exchange data with other processors
  - additional communication costs

### Implementation with Shift Operation

- beneficial for 1D stencils
- allow vectorization of data reads
- does not reduce memory traffic

### Implementation with tiles

- multidimensional stencils
- strip-mining (optimized for cache)
- example
  - two-dimensional array organized in row-by-row fashion
  - horizontal data in the same cache line, vertical far away
  - horizontal split
    - whole line does not fit cache, a lot of cache misses when accessing adjacent rows
  - vertical split
    - processors redundantly read the same cache line
  - strips (vertical)
    - each processor gets its strip of width equal to a multiple of cache line size
    - processing goes sequentially from top to bottom to maximize cache reuse
    - multiple of cache line size prevents false sharing between adjacent strips on output

      <img src="figures/stencil-2D.png" alt="2D stencil - vertical strips" width="40%" />

### Communication

- commonly the output of stencil is used as the input for the next iteration
  - double buffering
  - pointers to buffers are interchanged between iterations
- need for synchronization
- boundary regions (halo) of the grid may need explicit communication with neighboring processors
  - halo can be exchanged each iteration
  - data exchange can take place on each $k$-th iteration when halo radius is increased, and some redundant computation takes place on each processor
  - latency hiding (update of internal grid cells when waiting for halo exchange)

  <img src="figures/stencil-comm.png" alt="Border exchange strategies" width="90%" />

### Example: Heat distribution

- square surface, three edges touch boiling water, one edge is put in ice
- How is the heat distributed inside the surface?
- Laplace equation

  $\frac{\partial^2 T(x,y)}{\partial x^2} + \frac{\partial^2 T(x,y)}{\partial y^2} = 0$
  
- discretized Laplace equation is in proper form for iterative solving

  $T(x, y) = \frac{1}{4} \cdot (T(x-h, y) + T(x+h, y) + T(x, y-h) + T(x, y+h))$

- surface size ```N+2``` includes boundary values, tiles of size ```BLOCK_SIZE+2``` include halo

  <img src="figures/heat-GPU.png" alt="Heat surface arrangement" width="40%" />

- result

  <img src="figures/heat.png" alt="Heat distribution" width="50%" />

- solutions
  - [heat0.cu](files/1-stencil/heat0.cu): CPU reference code
  - [heat1.cu](files/1-stencil/heat1.cu): GPU code, reading directly from the global memory
  - [heat2.cu](files/1-stencil/heat2.cu): GPU code, using local memory, static allocation
    - first threads copy data to the local memory
      - local memory is organized as a 2D tile which includes halo needed by threads on edges to correctly compute the next iteration
      - each thread transfers the value for which it is responsible
      - threads on edges also transfer values from halo (cells)
    - threads can start with computation only when all data is in local memory
      - ```__syncthreads()``` represents a barrier in CUDA C
  - [heat3.cu](files/1-stencil/heat3.cu): GPU code, using local memory, dynamic allocation
  - [heat4.cu](files/1-stencil/heat4.cu): GPU code, using local memory, dynamic allocations
    - threads in a warp ar not accessing neighboring memory locations
    - degraded performance

## Reduce

- a collective operation
- reduce pattern allows data to be combined
  - combiner function $f(a, b) = a \oplus b$
  - pairwise operation
  - associativity must hold
    - $(a \oplus b) \oplus c = $a \oplus (b \oplus c)$
    - operands can be combined in any order
    - floating point addition and multiplication are only partially associative due to the limited precision
  - commutativity
    - $a \oplus b = b \oplus a$
    - not required, but enables additional reorderings
  - identity
    - initial value of reduction
- serial and parallel patterns

  <img src="figures/reduce-ser-par.png" alt="Serial and parallel reduce" width="75%" />

### Tiling

- use serial algorithm where possible
- do tree-like reduction to reduce communication costs

  <img src="figures/reduce-tiling.png" alt="Reduce with tiling" width="75%" />

- process
  - break the work to tiles
  - operate on tiles separately
  - combine partial results from tiles
- serial and tree algorithms
  - the number of combiner function applications is the same
  - serial algorithm requires less storage for intermediate results

### Theoretical Considerations

- sequential reduction of $N$ operands
  - $N−1$ reductions
  - each invocation of reduce function needs $\chi$ to complete
  - total execution time is $t_s = \chi (N-1)$
- parallel tree-like reduction, $n=2^k, k\in \mathbb{N}$
  - establishing communication takes $\lambda$ units of time
  - $N/2$ reductions in the first stage can go in parallel, $N/4$ in the second stage can go in parallel ... $1$ reduction in the last stage
  - altogether we have $\log_2 N$ stages with total $N-1$ reductions
  - total execution time equals $t_p = (\chi+\lambda)\log_2 N$

### Fusing Map and Reduce

- when map is feeding outputs directly into a reduction, the combination can be implemented more efficiently
- no need for synchronization between map and reduce stages
- no need to write intermediate results to memory or file
- map and reduce must be tiled in a same way

  <img src="figures/map-reduce.png" alt="Fused map reduce" width="60%" />

### Example: Dot Product

- vectors $\mathbf{a}$ and $\mathbf{b}$ of length $N$
- dot product

  $$\mathbf{a}\cdot \mathbf{b} = \sum_{i=0}^{N-1} a_i \cdot b_i $$

- solutions
  - [dotprod0.cu](files/2-reduce/dotprod0.cu): CPU reference code
  - [dotprod1.cu](files/2-reduce/dotprod1.cu): element-wise multiplication is performed on GPU, summation on CPU
  - [dotprod2.cu](files/2-reduce/dotprod2.cu): summation split between GPU and CPU
    - less data to transfer back to CPU
    - local memory to store products
    - serial summation of products from local memory inside a block thread

      <img src="figures/reduce-GPU.png" alt="Reduce on GPU - thread organization" width="90%" />

  - [dotprod3.cu](files/2-reduce/dotprod3.cu): tree-like summation of values inside a block, stride is increasing
    - in each stage the number of working threads inside a warp decreases

      <img src="figures/reduce-tree-inc.png" alt="Reduce on threads in a block, increasing stride" width="75%" />

  - [dotprod4.cu](files/2-reduce/dotprod4.cu): better warp managements
    - tree-like summation of values inside a block, stride is decreasing
    - all threads in a warp are working (up to the last warp)
    - at each stage the number of warps halves
    - considerably improved performance

      <img src="figures/reduce-tree-dec.png" alt="Reduce on threads in a block, decreasing stride" width="75%" />

  - [dotprod5.cu](files/2-reduce/dotprod5.cu): for threads inside the final warp we use light-weight barrier with ```__syncwarp()```
  - [dotprod6.cu](files/2-reduce/dotprod6.cu): generalization of solution to support block sizes not power of $2$
  - [dotprod7.cu](files/2-reduce/dotprod7.cu): trick to compute largest power of $2$ smaller or equal to the block size
  - [dotprod8.cu](files/2-reduce/dotprod8.cu): atomic add, no summation on CPU
  - [dotprod9.cu](files/2-reduce/dotprod9.cu): dynamic allocation of local memory
  - [dotprodA.cu](files/2-reduce/dotprodA.cu): managed memory solution

## Scan

- also prefix scan
- produces all partial reductions of an input sequence

  <img src="figures/scan-ser-inc.png" alt="Serial inclusive scan" width="45%" />

- exclusive and inclusive scan
- operation
  - input sequence:
    $[𝑎_0, 𝑎_1, 𝑎_2, ..., 𝑎_(N−1)]$
  - output:
    - exclusive scan: $[I, a_0, a_0 \oplus a_1, a_0 \oplus a_1 \oplus a_2, \ldots, a_0 \oplus \cdots \oplus a_{N-2}]$
    - inclusive scan: $[a_0, a_0 \oplus a_1, a_0 \oplus a_1 \oplus a_2, \ldots, a_0 \oplus \cdots \oplus a_{N-1}]$

  <img src="figures/scan-excl-incl.png" alt="Exclusive and inclusive scan" width="60%" />

- sequential approach
  - loop-carried dependence
- parallel approach
  - loop-carried dependence
  - similar to reduce
  - two solutions
    - Count on associativity of the combiner function ($\oplus$)
- combining scan
  - map
    - tiled scan: map can be applied before the first stage and/or after the last stage
    - reduces data transfers
  - reduce
    - similar scheme, can do both with little extra work

### Parallel scan with double buffering

- Hillis and Steele, 1986
- requires two buffers of length $N$

  <img src="figures/scan-double-buffering.png" alt="Parallel scan with double buffering" width="90%" />

### Two-phase parallel scan

- Blelloch, 1990
- tree-like approach
- requires only one buffer of length $N$
- more synchronizations
- first phase: up-sweep
  - stride increases: $0, \ldots, \lfloor\log_2 N\rfloor - 1$
  - see displacements of elements used in computation in figure below
- second phase: down-sweep
  - stride decreases: $\lfloor\log_2 N\rfloor - 1, \ldots, 0$
  - see displacements of elements used in computation in figure below

  <img src="figures/scan-two-phase.png" alt="Two-phase parallel scan" width="90%" />

### Tiled scan

- if problem cannot fit to one CPU, we split the problem to tiles
- three stage approach
  - each processor handles one tile (serial or parallel solution)
  - scan on last elements in each tile (blue) gives offsets (green)
  - addition of offsets can be again performed inside each tile to get final values (gray)

  <img src="figures/scan-tiles.png" alt="Parallel scan with tiles" width="90%" />

### GPU implementations

- tiled scan approach where size of a tile equals block size
- ```blockSum``` in GPU global memory holds the value of last element in each tile
- ```blockSum``` is a result of the ```scan``` kernel and is used by the ```add```kernel to compute offsets
- local memory
  - double length array presents input and output buffer
  - as there is no pointers in CUDA C, we use two displacements ```dIn``` and ```dOut``` to determine beginning of the input and output part of the buffer

  <img src="figures/scan-double-buffering-gpu.png" alt="Double buffering in local memory with CUDA C" width="30%" />

- solutions for inclusive scan
  - [scan0.cu](files/3-scan/scan0.cu): serial scan inside blocks (tiles) using local memory, serial computation of offsets
  - [scan1.cu](files/3-scan/scan1.cu): parallel double-buffering scan
    - double buffering inside a thread block (tile)
    - offsets are computed by reduce operation as each thread block has to do it on its own
    - addition of offsets goes in parallel
  - [scan2.cu](files/3-scan/scan2.cu): two-phase scan
    - only the ```scan```kernel is updated
