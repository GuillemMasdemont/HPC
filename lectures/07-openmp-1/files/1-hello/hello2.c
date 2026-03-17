// hello 1
// gcc -fopenmp -o hello1 hello1.c

#include <stdio.h>
#include "omp.h"

int main() {
    #pragma omp parallel 
	{
		#pragma omp sections
		// #pragma omp sections nowait 
		{
			#pragma omp section 
			{
				int ID = omp_get_thread_num();
				printf("hello(%d) ", ID);
			}
			#pragma omp section 
			{
				int ID = omp_get_thread_num();
				printf("world(%d) ", ID);
			}
		}

		#pragma omp sections 
		{
			#pragma omp section 
			{
				int ID = omp_get_thread_num();
				printf("HELLO(%d) ", ID);
			}
			#pragma omp section 
			{
				int ID = omp_get_thread_num();
				printf("WORLD(%d) ", ID);
			}
		}
    }
	return 0;
}
