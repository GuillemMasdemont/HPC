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
