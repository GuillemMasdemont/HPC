# Programming Patterns

## Pattern-based Programming

- Patterns are “best practices” for solving specific problems.
- Patterns are universal, can be applied to any programming system.
- Patterns can be used to organize your code, leading to algorithms that are more scalable and maintainable.
- A pattern supports a particular algorithmic structure with an efficient implementation.  
- Good parallel programming models support a set of useful parallel patterns with low-overhead implementations.
- New patterns can be derived from existing patterns.
- Serial and parallel patterns.

- Focus on
  - algorithm skeletons
    - building blocks, arrangement of tasks, data dependencies
    - focus on data parallelism to ensure scalability
  - implementation patterns (low-level, hardware specific)
    - granularity, good use of cache
  - Not design patterns (high level, abstract)

- Task
  - Task is a unit of independent (potentially parallel) work.
  - Tasks are executed by scheduling on software threads
  - Usually cooperative: at predicted switch points
  - Software threads are scheduled by OS onto hardware threads
  - Preemptive approach is most common (at any time)

## Serial Patterns

- Nesting
  - Fundamental compositional pattern
  - Allows for hierarchical composition
  - Any task block in a pattern can be replaced with a pattern with the same input and output configuration and dependencies
  - Nesting is crucial for structured, modular code

- Serial Control Flow Patterns  

  - Sequence

    ```C
    B = f(A);
    C = g(B);
    D = h(B, C);
    ```

    - No data dependencies
    - Data dependencies restrict the order of execution
    - Parallel generalization: superscalar sequence
      - removes code-text order constraint
      - Order tasks only by data dependencies
  - Selection (branch)

    ```C
    if (cond) {
      statementA;
    } else {
      statementB;
    }
    ```

    - Parallel generalization: speculative execution
      - ```cond```, ```statementA```, and ```statementB``` can be executed in parallel
      - ```statementA``` or ```statementB``` is discarded when ```cond```is known

  - Iteration

    ```C
    while (cond) {
      statement;
    }
    ```

    - Countable iteration (for loop) as a special case, important for parallel patterns.
    - Parallel generalization
      - The serial iteration pattern appears in several different parallel patterns: map, reduction, scan, recurrence, scatter, gather, pack
      - Parallel patterns have a fixed number of invocations, known in advance
      - Loop-carried dependencies limit parallelization
      - Problem of hidden data dependencies

        ```C
        for (int i = 0; i < n; i++)
          x[a[i]] = x[b[i]] * x[c[i]] + x[d[i]];
        ```

        ```C
        for (int i = 0; i < n; i++)
          y[a[i]] = x[b[i]] * x[c[i]] + x[d[i]];
        ```

        It seems that there is no dependencies in the second case. But if ```y``` points to the same location as ```x```, the dependencies are the same as in the first case.

    - recursion
      - dynamic nesting which allows functions to call themselves
      - stack memory allocation

  - serial data management patterns

    - random read and write
      - working with pointers, possible aliasing (two pointers refer to the same memory object)
      - aliasing can make vectorization and parallelization difficult
        - undefined result
        - extra object copies reduce performance
        - burden put on a programmer
      - working with array indices is safer and easier to transfer to another platform

    - stack allocation
      - efficient as arbitrary amount of data can be allocated and preserves locality
      - each thread gets its own stack

    - heap allocation
    Heap allocation
      - slower than stack allocation
      - allocation scattered all over memory (matrix allocation)
        - more expensive due to the memory hardware subsystem
      - implicitly sharing the data structure can lead to scalability problems
        - better is to maintain a separate memory pool for each worker and avoid global locks
        - gGet rid of false sharing
