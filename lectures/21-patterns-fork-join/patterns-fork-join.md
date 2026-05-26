# Patterns: Fork-join

- fork-join pattern can generate a high-level of parallelism
- serial divide-and-conquer algorithms can be efficiently parallelized with fork-join pattern
  - limits on speedup
  - most of the work should go deep into the recursion
- recursive approach to parallelism
  - need for work schedulers
- fork-join as a directed graph
  - control flow forks (divides) into multiple flows
    - one flow turns into more separate flows
    - each flow is independent and not constrained to do similar computation
  - multiple flows join (combine) the latter
    - after joining, only one flow continues
  - example: a task is split to two subtasks that executed in parallel and joined afterward

    <img src="figures/fork-join.png" alt="Fork-join pattern" width="25%">

## Divide-and-conquer

- typical divide-and-conquer scheme

    <img src="figures/divide-conquer.png" alt="Devide-and-conquer patternFork-join scheme" width="65%">

  - sub-problems must be independent

  ```C
  void divideAndConquer(problem P)
    if (P is base case)
      solve P;
    else {
        divide P into b sub-problems
        fork to conquer each sub-problem in parallel
        join
        combine sub-solutions into final solution
    } 
  ```

- $b$ sub-problems on $L$ levels permit up to $b^L$ parallel tasks
- the vast majority of work must be deep in the recursion where the parallelism is high
- gor good performance, it is crucial to select the proper size of the base case
  - the recursion should not go too deep as scheduling overheads will start to dominate
  - the operations before the fork and after the join should be fast, so they do not strangle speedup

## Fork-join Programming Model

- CUDA: patterns with a lot of control do not fit well with GPU concepts
- MPI: dynamic changing of the number of processors makes it possible; not used much as the number of processes on clusters must be determined at the job submission stage
- OpenMP: from version 3.0 onwards, it has explicit support for tasks

### OpenMP support for tasks

- tasks are independent units of work
- tasks can be nested
- tasks present additional level of abstraction
- tasks are sent to a pool of tasks
- OpenMP scheduler picks tasks from the pool and assigns them to software threads
  - tasks may be executed immediately
  - tasks may be deferred (for example waiting for task dependencies to be fulfilled)
  - schedulers are implementation specific, not determined by standard

- OpenMP syntax
  - `#pragma omp task`
    - indicates that subsequent statements can be independently forked as tasks
    - by default, variables are of type `firstprivate`
  - join with #pragma omp `taskwait`
    - waits for all tasks to join
  - tasks can only be used inside a parallel region
    - only one thread (master) starts the execution
  - example: function `divideAndConquer` and its call from the main routine

    ```C
    solution divideAndConquer(problem) {
        subproblem1 = fork(problem, 1);
        subproblem2 = fork(problem, 2);
        #pragma omp task
        solution1 = divideAndConquer(subproblem1)
        solution2 = divideAndConquer(subproblem2)
        #pragma omp task wait
        solution = join(solution1, solution2);
    }

    int main(void) {
        #pragma omp parallel
        #pragma omp master
        divideAndConquer(problem);
        return 0;
    }
    ```

  - programmer has some control on forking
    - `#pragma omp task final(condition)`
      - when condition executes to `true`, new tasks are not generated anymore
      - the computation is performed inside the calling task

### Recursive Implementation of the Map-reduce Pattern

- adaptive quadrature using trapezoidal rule
- compare quadrature on two levels:
  - area of trapezoid spanning from the lower to the upper bound (red)
  - sum of area of trapezoid spanning from the lower bound to the middle and the area of trapezoid spanning from the middle to the upper bound (green)

    <img src="figures/adapt-quad.png" alt="Adaptive wuadrature using trapezoidal rule" width="60%">

- if difference is grater than allowed, split interval to two halves and repeat quadrature on each halve
- code
  - [adaptquad_ser.c](files/adaptquad/adaptquad_ser.c) and [adaptquad_ser.sh](files/adaptquad/adaptquad_ser.sh): reference serial implementation
  - [adaptquad_par1.c](files/adaptquad/adaptquad_par1.c) and [adaptquad_par1.sh](files/adaptquad/adaptquad_par1.sh)
    - each task creates two sub-tasks
    - slow performance
  - [adaptquad_par2.c](files/adaptquad/adaptquad_par2.c) and [adaptquad_par2.sh](files/adaptquad/adaptquad_par2.sh)
    - added `final`clause to limit creation of parallel tasks deep in the recursion
  - [adaptquad_par3.c](files/adaptquad/adaptquad_par3.c) and [adaptquad_par3.sh](files/adaptquad/adaptquad_par3.sh)
    - each task creates only one sub-task
    - main tasks performs some computation on its own

## Choosing Base Cases

- when recursion goes to deep, scheduling overheads tend to swamp useful work
- two separate base cases at different levels
  - a base case for stopping parallel recursion
    - parallel task scheduling overheads
  - a base case for stopping serial recursion
    - function call overheads
    - less expensive compared to task scheduling overheads
  - serial recursion stops at much smaller problem sizes
- it is tempting to set the number of base cases equal to the number of parallel hardware threads
  - scheduler has no flexibility to balance load; even if problem is well balanced, operating system can cause issues
  - better is to over-decompose the problem and create some parallel slack

## Algorithm Complexity

- majority of divide-and-conquer problems can be described with relation

  $T(N) = at({n\over b}) + cn^d\quad, \quad T(1)=\mathrm{e}$

- task on level $l$ has $cn^d$ work itself and task on level $l+1$ has $ac(\frac{n}{b})^d$ work itself, leading to proportion
  $r = \frac{ac(\frac{n}{b})^d}{cn^d} = \frac{a}{b^d}$
- asymptotic solutions
  - case 1: $r > 1$: $T(n) = O(n^{\log_b a})$
    - the work exponentially increases with depth, bottom levels dominate
  - case 2: $r = 1$: $T(n) = O(n^d \log_2 n)$
    - the work at each level is about the same
    - the work is proportional to the work at top level times the number of levels
  - case 3: $r < 1$: $T(n) = O(n^d)$
    - the work exponentially decreases with depth, top levels dominate
- examples with $c = d = 1$

  <img src="figures/algorithm-complexity.png" alt="Divide-and-conquer algorithm complexity" width="80%">

### Karatsuba Polynomial Multiplication

- input: polynomials $a$ and $b$ of degree $n-1$ ($n$ coefficients)
- output: polynomial $c$ of degree $2n-2$ ($2n−1$ coefficients)
- the flat (high-school) method
  - concise and highly parallel for large $n$
  - creates $O(n^2)$ serial work
  - example:
    - $c(x) = a(x)b(x) = (a_1 x + a_0)(b_1 x + b_0) = a_1 b_1 x^2 + a_1 b_0 x + a_0 b_1 x + a_0 b_0$
    - four multiplications of coefficients
- idea of Karatsuba method
  - $c(x) = a(x)b(x) = (a_1 x + a_0)(b_1 x + b_0) = a_1 b_1 x^2 + (a_1 b_0 + a_0 b_1) x + a_0 b_0$
  - only three multiplications
    - $t_0=a_0b_0$
    - $t_2=a_1b_1$
    - $t_1=(a_1+a_0)(b_1+b_0)$
  - $c(x) = t_2x^2 + (t_1-t_0-t_2)x + t_0$
- each multiplication can be done by recursive application of Karatsuba method
  - for small polynomials flat method becomes more efficient
  - the method is commonly used for exact integer multiplication
  - example
    - $1234\times 5678$
      - $x=10$
      - $a_1(x) = 12 x^2 + 34 x^0$, $b_1(x) = 56 x^2 + 78$
      - $c_1(x) = a_1(x)b_1(x)$
        - $t_0 = 34 \times 78$
        - $t_2 = 12 \times 56$
        - $t_1 = (12 + 34)\times (56+78)$
        - it is still hard to calculate in a head, go one level deeper
        - calculation for $t_2$:
          - $a_2(x) = 1 x +2$, $b_2(x) = 5x+6$
          - $c_2(x) = a_2(x)b_2(x)$
            - $t_0 = 2 \times 6 = 12$
            - $t_2 = 1 \times 5 = 5$
            - $t_1 = (1+2)\times(5+6)=33$
            - $c_2(10) = 5 \times 10^2 + (33-12-5) \times 10 + 12 = 672$
        - following the same approach we also get $t_0 = 2652$ and $t_1=6164$ resulting in $c_1(10) = 7006652$
- code for polynomial multiplication
  - [pmult_ser.c](files/karatsuba/pmult_ser.c) and [pmult_ser.sh](files/karatsuba/pmult_ser.sh): reference implementation
  - [pmult_par.c](files/karatsuba/pmult_par.c) and [pmult_par.sh](files/karatsuba/pmult_par.sh): parallel recursive implementation
    - limiting the recursion depth (`CUTOFF`) is important to gain speedups

- algorithm time complexity
  - unlimited number of tasks
  - no early stopping of recursion
  - assumption $n=2^k$
  - serial algorithm: $T_1(n) = 3\cdot T_1(n/2) + n$
    - $T_1(2^k) = 2^k + 3T_1(2^{k-1}) = 2^k + 3^1 2^{k-1} + 3^2T_1(2^{k-2}) = ... = \sum_{i=0}^k 2^i 3^{k-i} =3^k \sum_{i=0}^k ({2\over 3})^i = 3^k\frac{1-(\frac{2}{3})^{k+1}}{1-\frac{2}{3}} = 3^{k+1}-2^{k+1}$
    - $T_1(2^k) = 3\cdot 3^{k}-2\cdot 2^{k} = 3(2^{\log_2 3})^k - 2\cdot 2^k = 3(2^k)^{\log_2 3} - 2\cdot 2^k$
    - $T_1(n) = 3n^{\log_2 3} - 2n$
  - parallel algorithm: $T_{\infty}(n)= 1\cdot T_{\infty}(n/2) + n$
    - $T_{\infty}(2^k) = 2^k + T_{\infty}(2^{k-1}) = 2^k + 2^{k-1} + T_{\infty}(2^{k-2}) = \sum_{i=0}^k 2^{k-i} = 2^k\sum_{i=0}^k (\frac{1}{2})^i = 2^k \frac{1-(\frac{1}{2})^{k+1}}{1-\frac{1}{2}} = 2^{k+1}-1$
    - $T_{\infty}(n) = 2n-1$
  - both equations follow the asymptotic solutions presented above
- algorithm space complexity
  - serial solution needs less memory as it can reuse structures: $M_1(n) = M_1(\frac{n}{2}) + O(n)\quad, \quad M_1(1) = O(1)$
  - parallel solution needs to store some temporary values
    - coefficients of $a(x)$, $b(x)$, and $c(x)$ can be used on all levels of recursion
    - additional memory is needed to store sum of coefficients ($a_1+a_0$ and $b_1+b_0$ in the example) and their product $t$
    - $M_{\infty}(n) = 3 M_{\infty}(\frac{n}{2}) + O(n)\quad, \quad M_{\infty}(1) = O(1)$

## Cache-oblivious Programming

- memory bandwidth constraints often limit speedup
- in such cases it is important to reuse data in cache
  - cache sizes vary among platforms, tailoring an algorithm to cache size becomes complicated
  - suboptimal solution which works well is cache-oblivious programming
    - cache paranoid programming
    - code is written to work well regardless of the actual cache structure
    - divide-and-conquer approach results in good data locality at multiple scales
      - with division the problem first fits to outermost cache
      - with further divisions it sooner or later fits to the inner cache

### Matrix Multiply-and-add

- $\mathbf{C}_{m \times n} = \mathbf{C}_{m \times n} + \mathbf{A}_{m \times k} \times \mathbf{B}_{k \times n}$
- similar to Strassen algorithm
- if matrices are small, use serial multiplication
- if matrices are large, divide multiplication to two parts
- the goal is to keep the matrices quadratic
  - always split the longest axis
    - when $m$ is the longest
        $\left[ \genfrac{}{}{0pt}{}{\mathbf{A}_0}{\mathbf{A}_1} \right] \times \mathbf{B} = \left[ \genfrac{}{}{0pt}{}{\mathbf{A}_0\times\mathbf{B}}{\mathbf{A}_1\times\mathbf{B}} \right]$
    - when $n$ is the longest
        $\left[\mathbf{A} \right] \times \left[ \mathbf{B}_0] \ \mathbf{B}_1 \right] = \left[ \mathbf{A}\times \mathbf{B}_0 \ \mathbf{A}\times \mathbf{B}_1 \right]$
    - when $k$ is the longest
        $\left[ \mathbf{A}_0 \ \mathbf{A}_1 \right] \times \left[ \genfrac{}{}{0pt}{}{\mathbf{B}_0}{\mathbf{B}_1} \right] = \left[ \mathbf{A}_0 \times \mathbf{B}_0 + \mathbf{A}_1 \times \mathbf{B}_1 \right]$
  - this way cache locality is maximized during multiplication
- pseudo code

  ```C
  // C[m][n] += A[m][k] * B[k][n]
  void MultiplyAdd(double **C, double **A, double **B) {
    if (less then CUTOFF operations are required to compute C) {
        MultiplyAddNonRecursive(C, A, B);
    }
    else if (n >= max(m, k)) {
        C = [C0 C1];
        B = [B0 B1];
        #pragma omp task
        MultiplyAdd(C0, A, B0);

        MultiplyAdd(C1, A, B1);
        #pragma omp taskwait
    }
    else if (m >= k) {
        C = [C0
             C1];
        A = [A0
             A1];
        #pragma omp task
        MultiplyAdd(C0, A0, B);

        MultiplyAdd(C1, A1, B);
        #pragma omp taskwait
    }
    else {
        A = [A0 A1];
        B = [B0
             B1];
        MultiplyAdd(C, A0, B0);
        MultiplyAdd(C, A1, B1);
    }
  }
  ```
  
- serial algorithm
  - time complexity: $T_1 = O(mnk)$
  - space complexity: $M_1 = O(mn) + O(mk) + O(kn)$
- parallel algorithm
  - time complexity: $T_{\infty} = \log_2 m + \log_2 n + k$
  - space complexity: $M_{\infty} = M_1$ with multiply-and-add
  - on the finest level each task is doing the multiplication of one element in matrix $\mathbf{C}$
  - the first two terms give the number of partitions
  - the last terms gives the time needed to get scalar product for one element of matrix $\mathbf{C}$
  - after partitioning, we can compute products in parallel
