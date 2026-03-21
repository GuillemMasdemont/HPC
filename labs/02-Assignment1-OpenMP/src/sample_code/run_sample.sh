#!/bin/bash

# Compile
gcc -O3 -lm -Xpreprocessor -fopenmp \
-I/usr/local/opt/libomp/include \
-L/usr/local/opt/libomp/lib \
-lomp parallel_seam_carving.c -o parallel_seam_carving

# Run
./parallel_seam_carving ../test_images/720x480.png parallel_out.png 128 1 sequential
