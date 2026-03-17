// Sieve of Eratosthenes - parallel version
// gcc -fopenmp -lm -o soe_par soe_par.c
// srun --reservation=fri --mem-per-cpu=2100 --threads-per-core=1 --cpus-per-task=8 soe_par 2000000000

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
        fprintf(stderr, "Bad arguments!\nUsage:\n\tsoe_par <max>\n\n");
        return 1;        
    }

    // reserve memory
    listSize = listMax - listMin + 1;
    list = (char *)calloc(listSize, sizeof(char));
    int listSupportSize = (int)sqrt(listMax);

    double timeStart = omp_get_wtime();
    #pragma omp parallel
    {
        char *listSupport = (char *)calloc(listSupportSize, sizeof(char));

        int procs = omp_get_num_threads();
        int id = omp_get_thread_num();        
        int myListMin = listMin + (double)listSize / procs * id;
        int myListMax = listMin + (double)listSize / procs * (id + 1) - 1;
        int myListSize = myListMax - myListMin + 1;

        // sieving
        int prime = 2;
        while (prime*prime <= myListMax) {
            // tag multiplies
            // listSupport
            int iStart = prime*prime - 2;    
            for (int i = iStart; i < listSupportSize; i += prime)
                listSupport[i] = 1;
            // list
            if (prime*prime > myListMin)
                iStart = prime*prime - myListMin;
            else if (myListMin % prime == 0)
                iStart = 0;     
            else
                iStart = prime - (myListMin % prime);
            int myOffset = myListMin - listMin;
            for (int i = iStart; i < myListSize; i += prime)
                list[myOffset + i] = 1;
            // find next prime
            while (listSupport[++prime-2]);
        }
        
        free(listSupport);
    }
    double timeEnd = omp_get_wtime();
    printf("Time: %lf\n", timeEnd - timeStart); 

    // write primes to file
    sprintf(filename, "soe_par.txt");
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
