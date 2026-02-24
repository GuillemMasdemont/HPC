# Introduction

## Why do we need high-performance computing?

- natural sciences:
  - everlasting pursuit of realism
  - simulations help us better understand nature
  - simulations are much more cost and time effective compared to real-world experiments
  - applications
    - weather forecasting, material sciences, nuclear physics,
    - chemistry, lattice QCD, biochemistry, life sciences, genomics, medicine

- data analytics
  - large quantities of data are collected
  - retrieving knowledge from data
  - confirmation of models
  - statistical and artificial intelligence modelling
  - applications:
    - high-energy physics, astrophysics,
    - artificial intelligence, deep learning
    - image and signal processing, computer vision, robotics

## Motivation

- ever growing need for enormous computing and data processing resources

- performance development over time: [top500.org](https://www.top500.org/statistics/perfdevel/)

- Europe
  - EuroHPC JU initiatives
  - HPC systems (petascale systems, exascale systems)
  - AI factories (19 + 2 giga)

- Slovenia
  - VEGA (IZUM, Maribor), 10 PFLOPS, 30 mio. EUR
  - new AI factory system (IZUM, Maribor), projected , 135 mio. EUR
  - goals
    - Upgrade existing HPC capabilities
    - Provide infrastructure for open research data
    - Data storage for Slovenian R&D
    - Offer computational capacities to industry
    - International cooperation
  - Arnes cluster
    - [hardware]((https://www.sling.si/en/arnes-hpc-cluster/)
    - software
      - SLURM batch system
      - AlmaLinux
      - user specific SW in modules
      - light virtualization (Apptainer)
  - Slovenia HPC fan club
    - Slovenian Environmental Agency
    - Institute Jožef Stefan
    - National Institute of Chemistry
    - Universities (UL FS, UL FRI, UL FE, UM)
  - HPC applications
    - examples:
      - CERN: CMS, ATLAS
      - OpenFOAM, ANSYS
      - Gromacs, Quantum espresso
      - AI: Keras, TensorFlow, Keras, PyTorch

- reasons for HPC
  - faster time to solution (response time)
  - solve bigger computing problems (in the same time)

- motivation for parallel processing
  - effective use of machine resources
  - cost efficiencies
  - overcoming memory constraints
  - HPC = parallel HW + concurrency + performance

- our approach
  - How does my application scale?
  - I have a specific problem and no adequate SW package. How to solve it more efficiently?
  - How to develop such nice tools and packages?
  - There is a new technology available.
    - Can my problem make use of it?
    - How to adapt my program for it?
  - Working on a cluster during the course
    - the course topic directly addresses clusters
    - we all use the same HW (multi-core nodes with GPUs)
    - the system is well-maintained by professional admins (HW, SW installation)
    - you can access the cluster from anywhere
    - you have valid access for the whole school year, you can use the systems for other courses as well (without reservation), for extension write to support@sling.si 

## Pervasive parallelism

- von Neumann architecture: CPU, memory, I/Oß
- for a long time (until 2004), the performance of computer systems increased through miniaturization (relays, vacuum tubes, increasingly smaller and more numerous transistors, raising the clock frequency) and improvements in hardware.
- Dennard scaling applied: reducing the size of a transistor by half increases their number fourfold, each transistor operates twice as fast (shorter connections), while the amount of dissipated heat remains unchanged
- characteristics of processors over time ([graph](https://www.karlrupp.net/2018/02/42-years-of-microprocessor-trend-data/))
- Moore’s law: system performance doubles every 18 months ([graf](https://en.wikipedia.org/wiki/Transistor_count#/media/File:Moore's_Law_Transistor_Count_1970-2020.png))
- a problem arises with heat dissipation when processor power consumption exceeds 130 W
- multi‑core architectures begin to appear after 2004, which are more energy‑efficient
- by increasing the number of processor cores, the validity of Moore’s law is maintained

## Multi-core processors are more energy efficient than single-core ones

- a processor’s power consumption depends on the clock frequency (f), supply voltage (U), and the circuit’s capacitance (C)
- a higher frequency means more switching events and therefore higher energy consumption
- if pushed too far, chips overheat and begin to operate unreliably

    $$ P = U_0I=U_0\frac{de}{dt}=U_0C\frac{dU}{dt}$$

    $$ U = U_0\sin(2\pi f t) $$

    $$ \frac{dU}{dt} = 2\pi fU_0\cos(2\pi f t)$$

    $$ P = 2\pi C f U_0^2 \cos(2\pi f t)$$

    $$ P = k C f U_0^2 $$

- two processor cores can operate at a lower voltage, but they have higher capacitance
- with two processor cores, we can perform approximately the same amount of work at half the clock frequency

  <img src="figures/energy-consumption-single-double.core.png" alt="energy consumption: single- and double-core systems" width="75%">

## Three walls in 2025

key limitations that led to multi‑core processors:

- power consumption limits
- limits of parallelism inside a single processor
  - pipelining
  - speculative execution
  - super-scalar processors
- memory bandwidth limitations
  - processors are much faster than memory
  - caches help, but things get complicated on multi‑core processors
  - scalability problems
- support for developing parallel software
  - despite many years of research, automatic parallelization does not work well
  - compilers capable of automatically converting our sequential programs into parallel ones are still in development and perform poorly
  - libraries that support parallelization
  - new programming languages with built‑in support for parallelism
- solution
  - to make good use of new architectures, we must write parallel programs ourselves
  - new processors include mechanisms that can exploit parallel programs (hardware threads)
  - to achieve good results, we must understand the architecture well
  - how to write code that will remain efficient on future processors
