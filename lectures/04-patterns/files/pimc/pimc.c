#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

int main(int argc, char* argv[]) 
{  
	long	i, shots, hits;
	double	x, y, pi;
    char   fileName[100];
	FILE	*stream;

    hits = 0;
    shots = atoi(argv[1]);

    if (argc < 3 || shots == 0) {
        fprintf(stderr, "No arguments or bad arguments!\nUsage:\n\tpimc <mi. of shots> <jobid>\n\n");
        return 1;        
    }

	shots *= 1000000;
    int jobid = 1 + atoi(argv[2]);
    srand(time(NULL)*jobid);

    for(i = 0; i < shots; i++) {
        x = rand()/(double)RAND_MAX;
        y = rand()/(double)RAND_MAX;

        if(x*x+y*y < 1)
            hits++;
    }

    pi = 4.0*hits/(double)shots;

    sprintf(fileName, "pi-%s.out", argv[2]);
    if( (stream = fopen(fileName, "w")) != NULL ) {
        fprintf(stream, "%ld\t%ld\n", shots, hits);
        fprintf(stream, "%1.16lf\n", pi);
        fclose(stream);
    }

    return 0;
}
