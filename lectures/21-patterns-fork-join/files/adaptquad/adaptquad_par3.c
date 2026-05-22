// adaptive quadrature (numerical integration) of function func
// compile as: gcc adaptquad_par3.c -fopenmp -lm -o adaptquad_par3

#include <stdio.h>
#include <math.h>
#include <omp.h>

#define TOLERANCE 1e-14
#define GRANULARITY 64      // 64 = 2^6 --> 6 levels of threading

double func(double x) {
    return exp(-x*x);
}

double quad_par(double (*f)(double), double lower, double upper, double tol) {
// function to integrate, lower bound, upper bound, allowed error
    double quad;            // result of quadrature
    double h;               // interval size
    double middle;          // middle of interval
    double quad_coarse;     // coarse-grained approximation (one trapezoid)
    double quad_fine;       // fine-grained approximation (two trapezoids)
    double quad_lower;      // quadrature of lower interval 
    double quad_upper;      // quadrature of upper  interval
    double eps;             // difference in approximations

    h = upper - lower;
    middle = (lower + upper) / 2;

    // check the quality of solution 
    quad_coarse = h * (f(lower) + f(upper)) / 2.0;
    quad_fine = h/2 * (f(lower) + f(middle)) / 2.0 + h/2 * (f(middle) + f(upper)) / 2.0;
    eps = fabs(quad_coarse - quad_fine);

    if (eps > tol) {
        #pragma omp task shared(quad_lower) final(GRANULARITY*tol < TOLERANCE)
        quad_lower = quad_par(f, lower, middle, tol / 2);

//        #pragma omp task shared(quad_upper) final(GRANULARITY*tol < TOLERANCE)
        quad_upper = quad_par(f, middle, upper, tol / 2);

        #pragma omp taskwait
        quad = quad_lower + quad_upper;
    }
    else
        quad = quad_fine;

    return quad;
}

int main(int argc, char* argv[]) {
    double quadrature;
    double dt = omp_get_wtime();

    #pragma omp parallel
    #pragma omp master
    quadrature = quad_par(func, -4.0, +4.0, TOLERANCE);

    dt = omp_get_wtime() - dt;

    printf("Integral: %lf\nTime: %lf\n", quadrature, dt);

    return 0;
}
