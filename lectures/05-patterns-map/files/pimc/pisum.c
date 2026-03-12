#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) 
{  
    int     i, numOfFiles;
	long    myShots, myHits, shots, hits;
	double	pi;
    char   fileNameIn[100];
	
    FILE	*streamIn, *streamOut;

    hits = 0;
    shots = 0;
    
    numOfFiles = atoi(argv[1]);

    if (argc < 2 || numOfFiles == 0) {
        fprintf(stderr, "No arguments or bad arguments!\nUsage:\n\tpisum <num. of files>\n\n");
        return 1;        
    }

    for(i=0; i<numOfFiles; i++) {
        sprintf(fileNameIn, "pi-%d.out", i);
        if( (streamIn = fopen(fileNameIn, "r")) != NULL ) {
            fscanf(streamIn, "%ld\t%ld", &myShots, &myHits);
            fclose(streamIn);
            shots += myShots;
            hits += myHits;
        }
        else {
            fprintf(stderr, "Error reading input files...\n");
            return 1;
        }        
    }

    pi = 4.0*hits/(double)shots;
    if( (streamOut = fopen("pi-all.out", "w")) != NULL ) {
        fprintf(streamOut, "%ld\t%ld\n", shots, hits);
        fprintf(streamOut, "%1.16lf\n", pi);
        fclose(streamOut);
    }

    return 0;
}
