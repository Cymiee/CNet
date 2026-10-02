#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)


__global__ void add(float *a, float *b, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *h_inputa, *h_inputb, *h_output;
    int n = 1000;
    size_t size = n * sizeof(float);

    h_inputa = (float *)malloc(size);
    h_inputb = (float *)malloc(size);
    h_output = (float *)malloc(size);
    for (int i = 0; i < n; ++i) {
        h_inputa[i] = (float)(rand() % 20 - 10);
        h_inputb[i] = (float)(rand() % 20 - 10);
    }

    float *d_inputa, *d_inputb, *d_output;
    CUDA_CHECK(cudaMalloc(&d_inputa, size));
    CUDA_CHECK(cudaMalloc(&d_inputb, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaMemcpy(d_inputa, h_inputa, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inputb, h_inputb, size, cudaMemcpyHostToDevice));
    add<<<numBlocks, threadsPerBlock>>>(d_inputa, d_inputb, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    float *expected_output = (float *)malloc(size);
    for (int i = 0; i < n; ++i) {
        expected_output[i] = h_inputa[i] + h_inputb[i];
    }
    int correct = 1;
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_output[i] - expected_output[i]) > 1e-5f) { correct = 0; break; }
    }

    printf(correct ? "Add matches CPU reference - PASS\n" : "Add FAIL\n");

    free(h_inputa);
    free(h_inputb);
    free(h_output);
    free(expected_output);
    CUDA_CHECK(cudaFree(d_inputa));
    CUDA_CHECK(cudaFree(d_inputb));
    CUDA_CHECK(cudaFree(d_output));
}