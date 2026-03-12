#include "omp.h"
#include <stdio.h>
#include <unistd.h>

int main(void)
{
	int i;

	// #pragma omp parallel for schedule(static, 1)
	// #pragma omp parallel for schedule(static, 2)
	// #pragma omp parallel for schedule(dynamic, 1)
	// #pragma omp parallel for schedule(dynamic, 2)
	// #pragma omp parallel for schedule(guided, 1)
	for(i=0; i<10; i++)
	{
		int id = omp_get_thread_num();
		if(i==0)
			usleep(1000000);
		printf("Iteration %i done by thread %d.\n", i, id);
	}

	return 0;
}
