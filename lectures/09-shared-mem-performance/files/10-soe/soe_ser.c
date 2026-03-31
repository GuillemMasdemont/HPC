// Sieve of Eratosthenes - serial version
// gcc -fopenmp -lm -o soe_ser soe_ser.c
// srun --reservation=fri --mem-per-cpu=2100 --threads-per-core=1 --cpus-per-task=1 soe_ser 2000000000

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>    

int main(int argc, char* argv[]) 
{  
    char *list;
    int listSize;
    int listMin, listMax;
    int count;
    char filename[100];
    FILE *f;

    listMin = 2;
    if (argc == 2)
        listMax = atol(argv[1]);
    if (argc != 2 || listMax == 0 || listMax < listMin) {
        fprintf(stderr, "Bad arguments!\nUsage:\n\tsoe_ser <max>\n\n");
        return 1;        
    }
    
    // reserve memory
    listSize = listMax - listMin + 1;
    list = (char *)calloc(listSize, sizeof(char));

    double timeStart = omp_get_wtime();
    // sieving
    int prime = 2;
    while (prime*prime <= listMax) {
        // tag multiplies
        int iStart = prime*prime - listMin;
        for (int i = iStart; i < listSize; i += prime)
            list[i] = 1;
        // find next prime
        while (list[++prime-2]);
    }
    double timeEnd = omp_get_wtime();
    printf("Time: %lf\n", timeEnd - timeStart); 
    
    // write primes to file
    sprintf(filename, "soe_ser.txt");
    if( (f = fopen(filename, "w")) != NULL ) {
        fprintf(f, "Time: %lf\n\n", timeEnd - timeStart); 
        count = 0;
        for(int i = 0; i < listSize; i++)
            if(!list[i]) {
                // fprintf(f, "%ld\n", i+listMin);
                count++;
            }
        fprintf(f, "----------\ninterval[%ld, %ld]\n%ld\n", listMin, listMax, count);
        fclose(f);
    }

    // free memory
    free(list);

    return 0;
}
