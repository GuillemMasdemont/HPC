#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"

int simple_mult_par(double *c, double *a, double *b, int n) {
    #pragma omp parallel for
    for (int i = 0; i < 2*n-1; i++)
        c[i] = 0;
    
    #pragma omp parallel
    for (int i = 0; i < n; i++) {
        #pragma omp for
        for(int j = 0; j < n; j++)
            c[i+j] += a[i]*b[j];
    }
}

int simple_mult(double *c, double *a, double *b, int n) { 
    for (int i = 0; i < 2*n-1; i++)
        c[i] = 0;
    for (int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            c[i+j] += a[i]*b[j];
}

int karatsuba_mult_par(double *c, double *a, double *b, int n, int cutoff) {
    if (n <= cutoff)
        simple_mult(c, a, b, n);
    else {
        int m = n/2;
        double *a_ = malloc((4*(n-m)-1)*sizeof(double));    // a_ = aL + aH
        double *b_ = &a_[n-m];                              // b_ = bL + bH
        double *t = &b_[n-m];                               // t = tLH = a_ * b_

        // reset c
        c[2*m-1] = 0;                   // all other elements are assigned in next two calls
        #pragma omp task
        // c[0] = tL = aL*bL
        karatsuba_mult_par(&c[0], &a[0], &b[0], m, cutoff);
        // c[2*m] = tH = aH*bH
        #pragma omp task
        karatsuba_mult_par(&c[2*m], &a[m], &b[m], n-m, cutoff);

        for (int i = 0; i < m; i++) {
            a_[i] = a[0+i] + a[m+i];    // a_ = aL+aH
            b_[i] = b[0+i] + b[m+i];    // b_ = bL+bH
        }
        for (int i = m; i < n-m; i++) {
            a_[i] = a[m+i];             // aH and bH can have one additional element
            b_[i] = b[m+i];             
        }

        // tLH = aLH*bLH
        karatsuba_mult_par(t, a_, b_, n-m, cutoff); // n-m >= m; n-m = m, or m+1
        #pragma omp taskwait

        // t = tLH - tL - tH
        for (int i = 0; i < 2*m-1; i++)
            t[0+i] -= c[0+i] + c[2*m+i];
        for (int i = 2*m-1; i < 2*(n-m)-1; i++)
            t[0+i] -= c[2*m+i];        
        // c = tL + tH --> c = tL + tH + tLH
        for (int i = 0; i < 2*(n-m)-1; i++)
            c[m+i] += t[0+i];

        free(a_);
    }
}

int main(int argc, char * argv[]) {
    int N, CUTOFF;
    double *a, *b, *cs, *ck;

    if (argc != 3) {
        printf("%s <polynomial degree> <cutoff degree>\n", argv[0]);
        return 1;
    }
    else {
        N = atoi(argv[1]);
        CUTOFF = atoi(argv[2]);
    }

    a = (double *)malloc(N*sizeof(double));
    b = (double *)malloc(N*sizeof(double));
    cs = (double *)malloc((2*N-1)*sizeof(double));
    ck = (double *)malloc((2*N-1)*sizeof(double));

    for (int i = 0; i < N; i++)
        a[i] = b[i] = i+1;

    double stime = omp_get_wtime();
    simple_mult_par(cs, a, b, N);
    stime = omp_get_wtime() - stime;

    double ktime = omp_get_wtime();
    #pragma omp parallel
    #pragma omp master
    karatsuba_mult_par(ck, a, b, N, CUTOFF);
    ktime = omp_get_wtime() - ktime;

    int unequal = 0;
    for (int i = 0; i < 2*N-1; i++) {
        if (fabs(ck[i]-cs[i]) > 1e-5) {
            unequal++;
            printf("%.0f %.0f\n", cs[i], ck[i]);
        }
    }
    printf("\nUnequal = %d\n", unequal);
    printf("time: simple: %f, Karatsuba: %f\n", stime, ktime);

    free(a);
    free(b);
    free(cs);
    free(ck);

    return 0;
}
