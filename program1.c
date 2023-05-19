#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define M 1000
#define N 1000
#define K 1000

int A[M][K];
int B[K][N];
int C[M][N];

int num_threads = 2;

void *matrix_multiplication(void *arg)
{
    int thread_id = *(int *)arg;
    int start = (thread_id * M) / num_threads;
    int end = ((thread_id + 1) * M) / num_threads;

    for (int i = start; i < end; i++)
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
    return NULL;
}

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

    pthread_t threads[num_threads];
    int thread_id[num_threads];

    for (int i = 0; i < num_threads; i++)
    {
        thread_id[i] = i;
        pthread_create(&threads[i], NULL, matrix_multiplication, &thread_id[i]);
    }

    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
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