/* 
	Dining Philosophers Problem
	https://en.wikipedia.org/wiki/Dining_philosophers_problem	
	Five philosophers, plate of spaghetti and five forks.	
	
	Philosophers have a discussion: they think and talk, become hugry, eat, think and talk, ...
	Each philosopher eats with two forks, he can only take forks of his neighbor

	no control

	srun --reservation=fri --cpus-per-task=5 phil-0
*/

#include <stdio.h>
#include <unistd.h>
#include <omp.h> 

#define P 		5
#define COURSES	50

void discussion(void) {
	int p;
	int course = 0;

	p = omp_get_thread_num();
	printf ("P%d joins the discussion.\n", p);

	while (course < COURSES) {
		printf("P%d is thinking and talking.\n", p);
		usleep(10000*(p+1));
		printf("P%d is hungry.\n", p);
		fflush(stdout);

		usleep(1000);
		
		printf("P%d is eating course %d.\n", p, course);
		fflush(stdout);
		usleep(10000*(p+1));
		printf("P%d finished with course %d.\n", p, course);
		course++;
	}

	printf("P%d leaves the discussion.\n", p);
}

 
int main(void) {
	int p;

	omp_set_num_threads(P);

	#pragma omp parallel for
	for (p = 0; p < P; p++)
		discussion();
		
	return 0;
}
