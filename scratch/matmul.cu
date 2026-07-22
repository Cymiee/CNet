#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)


__global__ void matmul(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    int M = 512, K = 512, N = 512;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    h_A = (float *)malloc(size_A);
    h_B = (float *)malloc(size_B);
    h_C = (float *)malloc(size_C);

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = (float)(rand()/RAND_MAX);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (float)(rand()/RAND_MAX);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    float *expected = (float *)malloc(size_C);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[row * K + k] * h_B[k * N + col];
            }
            expected[row * N + col] = sum;
        }
    }
    int correct = 1;
    for (int i = 0; i < M * N; ++i) {
        if (fabsf(h_C[i] - expected[i]) > 1e-3f) { correct = 0; break; }
    }
    printf(correct ? "Matmul PASS\n" : "Matmul FAIL\n");
    free(expected);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
}
