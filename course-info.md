# Course info

## Objectives

- Get the theoretical and practical knowledge from the areas of parallel and distributed systems, parallel programming and processing, to effectively tackle computational problems using modern computing platforms and tools.  
- Parallelize scientific and engineering problems by structuring them correctly, selecting the appropriate hardware, and applying suitable programming concepts to produce efficient solutions.  
- Gain knowledge to work with the national high-performance infrastructure.
- Gained programming skills:
  - shared memory programming with OpenMP,
  - modern accelerator programming with CUDA,
  - distributed memory programming with OpenMP.
- Parallel programming patterns:
  - Parallel programming from a point of view of parallel computation patterns.
  - Map, Reduce, Stencil and recurrence, Collectives, Data reorganization, Fork-Join, Pipeline.

## Outcomes

- Hands-on experience with modern parallel computing architectures.
- Knowledge of choosing a proper hardware architecture to speed up an algorithm at hand.
- Ability to transform algorithms from the computational area to efficient programming code for modern computing architectures.
- Skills to compute and manage data on large-scale national infrastructural systems.
- Skills to analyze code with respect to performance and suggest and implement performance improvements.

## Topics covered

1. Fundamental concepts of high-performance computing 
2. Introduction to national high-performance computing infrastructure
3. Shared-memory programming
    - concurrency, threads, tasks
    - synchronization, scheduling
    - vectorization
    - using multithreading libraries
4. GPU accelerated programming
    - offload programming model
    - processor and memory hierarchy,
    - thread organization
    - robust and efficient GPU kernel programming and execution 
    - programming tensor cores
    - heterogeneous programming
5. Distributed-memory programming
    - message-passing paradigm
    - point-to-point and collective communication
    - using message-passing libraries
6. Parallel programming patterns
    - pipeline, functional and data parallelism
    - map, reduce, stencil, scan, fork-join, data reorganization patterns
7. Performance
    - speedup, efficiency, scalability
    - Amdahl and Gustafson-Barsis laws
    - load balancing
    - optimization and profiling
8. Case studies
    - cellular automata, differential equation solvers, N-body problems
    - dense and sparse matrix multiplication, sorting

## Tools

Throughout the course we will use

- a supercomputer,
- Linux operating system,
- C programming language along with necessary libraries
- VSCode development environment 

## Literature

- [IPP] P.S. Pacheco, M. Malensek. An Introduction to Parallel Programming, 2nd Edition, Morgan Kaufman, 2022
- [SPP] M. McCool, A. D. Robinson, J. Reinders. Structured Parallel Programming, Morgan Kaufmann, 2013
- [IPC] R. Trobec, B. Slivnik, P. Bulić, B. Robič. Introduction to Parallel Computing, Springer, 2018
- [PP] M. J. Quinn. Parallel Programming in C with MPI and OpenMP. Mc Graw Hill, 2003
- [IHPC] V. Eijkhout et al. Introduction to High-Performance Scientific Computing, Creative Commons, 2015
- [HC] B.R. Gaster et. al. Heterogeneous Computing with OpenCL. Morgan Kaufmann, 2013

## Staff

- lecturer: [Uroš Lotrič](https://fri.uni-lj.si/en/about-faculty/employees/uros-lotric)
- assistant: [Davor Sluga](https://fri.uni-lj.si/sl/o-fakulteti/osebje/davor-sluga)

## Grading

### Lectures

We encourage you to attend the lectures, where you will get acquainted with the basic concepts and principles, which are then used in the lab assignments. During the labs, it is assumed that you already have all the necessary knowledge from the lectures.

### Labs

You will get 4 assignments during the semester. You will work in pairs. You must submit your solutions through [ucilnica](https://ucilnica.fri.uni-lj.si/hpc) and present your solution during labs.

### Final grade

The final grade is computed as an average of grades received for labs (50%) and written exam (50%).

Requirements for a positive grade

- at least 3 lab assignments done,
- at least 50% of the total points achieved from the assignments,
- at least 50% of the total points achieved on the written exam.
