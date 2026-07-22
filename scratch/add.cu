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

    h_inputa = (float *)malloc(n * sizeof(float));
    h_inputb = (float *)malloc(n * sizeof(float));
    h_output = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        h_inputa[i] = (float)(rand() % 20 - 10);
        h_inputb[i] = (float)(rand() % 20 - 10);
    }

    float *d_inputa, *d_inputb, *d_output;
    CUDA_CHECK(cudaMalloc(&d_inputa, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inputb, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));

    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaMemcpy(d_inputa, h_inputa, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inputb, h_inputb, size, cudaMemcpyHostToDevice));
    add<<<numBlocks, threadsPerBlock>>>(d_inputa, d_inputb, d_output, n);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    //check if the output is correct
    float *expected_output = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        expected_output[i] = h_inputa[i] + h_inputb[i];
    }
    int correct = 1;
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_output[i] - expected_output[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }

    if (correct) {
        printf("Add kernel executed successfully!\n");
    } else {
        printf("Add kernel failed!\n");
    }

    free(h_inputa);
    free(h_inputb);
    free(h_output);
    free(expected_output);
    CUDA_CHECK(cudaFree(d_inputa));
    CUDA_CHECK(cudaFree(d_inputb));
    CUDA_CHECK(cudaFree(d_output));
}
