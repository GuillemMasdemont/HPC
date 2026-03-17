#include <stdio.h>
#include <stdlib.h>
#include <time.h>

char** board_initialize(int n, int m)
{
	int k, l;

	char* bd = (char*)malloc(sizeof(char) * n*m);
	char** b = (char**)malloc(sizeof(char*) * n);
	for (k = 0; k < n; k++)
		b[k] = &bd[k * m];

	for (k = 0; k < n; k++)
		for (l = 0; l < m; l++)
			b[k][l] = rand() < 0.25 * RAND_MAX;

	return b;
}

void board_update(char*** b, char*** bn)
{
	char** bt;
	bt = *b;
	*b = *bn;
	*bn = bt;
}

void board_free(char** b)
{
	free(*b);
	free(b);
}

void board_print(char** b, int n, int m)
{
	int k, l;

	for (k = 0; k < n; k++)
	{
		for (l = 0; l < m; l++)
			printf("%d", b[k][l]);
		printf("\n");
	}
	printf("\n");
}

int count_neighbors(char** b, int n, int m, int i, int j)
{
	int di, dj;
	int sum = 0;

	for (di = -1; di < 2; di++)
		for (dj = -1; dj < 2; dj++)
			if (di != 0 || dj != 0)
				sum = sum + b[(i + di + n) % n][(j + dj + m) % m];

	return sum;
}

