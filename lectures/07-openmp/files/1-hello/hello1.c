// hello 1
// gcc -fopenmp -o hello1 hello1.c
// srun --cpus-per-task=8 hello1

#include <stdio.h>
#include <unistd.h>
#include "omp.h"

int main() {
    #pragma omp parallel
	{
		int ID = omp_get_thread_num();
		printf("hello(%d) ", ID);
		usleep(1);
		printf("world(%d) \n", ID);
    }
	return 0;
}
