#include <stdio.h>
#include <stdlib.h>
#include "/opt/homebrew/Cellar/libomp/16.0.0/include/omp.h"
#include <time.h>

#define M 1000
#define N 1000
#define K 1000

int A[M][K];
int B[K][N];
int C[M][N];

int num_threads = 2;

int main()
{

    int value = 1;

    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k++)
        {
            A[m][k] = value++;
        }
    }

    value = 1;

    for (int k = 0; k < K; k++)
    {
        for (int n = 0; n < N; n++)
        {
            B[k][n] = value++;
        }
    }

    double time_spent = 0.0;

    clock_t begin = clock();

#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < K; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    clock_t end = clock();

    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

    printf("The elapsed time is %f seconds\n", time_spent);

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}