#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)
    do {
        cudaError_t err = call;
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrprString(err));
            exit(EXIT_FAILURE);
        }
    } while (0)

// THE KERNEL FUNCTION
// Marked __global__ = runs on the GPU and can be called from CPU. Returns void.
__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000;
    size_t bytes = n * sizeof(float);

    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);
    float *c = (float *)malloc(bytes);

    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)i * 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;

    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    add<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i+=20) {
        printf("%s", c[i] == 3.0f * (float)i ? "PASS\n" : "FAIL\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
}