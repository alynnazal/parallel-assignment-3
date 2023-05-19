#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define M 1024
#define N 1024
#define K 1024

_global_ void matrixMul(float *a, float *b, float *c, int m, int n, int k)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && j < n)
    {
        float sum = 0;
        for (int l = 0; l < k; l++)
        {
            sum += a[i * k + l] * b[l * n + j];
        }
        c[i * n + j] = sum;
    }
}

double getTime()
{
    struct timeval tim;
    gettimeofday(&tim, NULL);
    return tim.tv_sec + (tim.tv_usec / 1000000.0);
}

int main()
{
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int size = M * K * sizeof(float);

    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    for (int i = 0; i < M * K; i++)
        a[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++)
        b[i] = rand() / (float)RAND_MAX;

    cudaMalloc((void **)&d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    size = K * N * sizeof(float);

    cudaMalloc((void **)&d_b, size);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    size = M * N * sizeof(float);

    cudaMalloc((void **)&d_c, size);

    dim3 dimBlock(32, 32);
    dim3 dimGrid((N - 1) / dimBlock.x + 1, (M - 1) / dimBlock.y + 1);

    double startTime = getTime();
    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N, K);
    double endTime = getTime();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    double elapsedTime = endTime - startTime;
    printf("Elapsed Time: %.6f seconds\n", elapsedTime);

    printf("Resulting Matrix:\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.2f ", c[i * N + j]);
        }
        printf("\n");
    }

    free(a);
    free(b);
    free(c);

    return 0;
}
