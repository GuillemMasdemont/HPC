// Mandelbrot set calculation using OpenMP
// gcc -fopenmp -o mb1 mb1.c
// srun --cpus-per-task=16 mb1

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int mandelbrot(double cx, double cy, int max_iter) {
    int iter = 0;
    double x = 0.0, y = 0.0;    // z = x + iy
    double xNew = 0.0;
    while (x*x + y*y <= 4.0 && iter < max_iter) {
        xNew = x*x - y*y + cx;
        y = 2.0 * x * y + cy;
        x = xNew;
        iter++;
    }
    return iter;
}

int main(int argc, char *argv[]) {

    const int iterMax = 1000;
    const double xMin = -2.5, xMax = 1.0;
    const double yMin = -1.0, yMax = 1.0;

    const int width = 1920, height = 1080;

    double dx = (xMax - xMin) / (width - 1);
    double dy = (yMax - yMin) / (height - 1);

    double timeStart = omp_get_wtime();
    #pragma omp parallel for
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            double cx = xMin + i * dx;
            double cy = yMax - j * dy;
            int iter = mandelbrot(cx, cy, iterMax);
            int grayColor = (int)(255.0 * iter / iterMax);
        }
    }
    double timeEnd = omp_get_wtime();
    printf("Time taken: %f seconds\n", timeEnd - timeStart);

    return 0;
}
