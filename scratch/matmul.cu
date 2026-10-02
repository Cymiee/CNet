#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE 16

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Check a kernel launch (which CUDA_CHECK can't wrap, since <<<>>> has no return).
// Call right after a launch. Catches silent launch/exec failures (e.g. wrong -arch).
#define CHECK_KERNEL()                                                     \
    do {                                                                   \
        cudaError_t launchErr = cudaGetLastError();                        \
        if (launchErr != cudaSuccess) {                                    \
            fprintf(stderr, "Kernel launch error at %s:%d - %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(launchErr));    \
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

__global__ void matmul_tiled(float *A, float *B, float *C, int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < K / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * K + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


int main() {
    int M = 512, K = 512, N = 512;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / (float)RAND_MAX;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE, TILE);
    dim3 numBlocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // CPU reference
    float *expected = (float *)malloc(size_C);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += h_A[row * K + k] * h_B[k * N + col];
            expected[row * N + col] = sum;
        }
    }

    const float EPS = 1e-2f;   // looser: sums of 512 products accumulate rounding

    // ---- NAIVE ----
    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);   // warmup
    CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    CHECK_KERNEL();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // diagnostic on element 0
    printf("sample: h_A[0]=%f h_B[0]=%f  cpu[0]=%f  gpu[0]=%f\n",
           h_A[0], h_B[0], expected[0], h_C[0]);

    int naive_ok = 1;
    for (int i = 0; i < M * N; ++i) {
        if (fabsf(h_C[i] - expected[i]) > EPS) {
            printf("naive mismatch at %d: gpu=%f cpu=%f diff=%f\n",
                   i, h_C[i], expected[i], fabsf(h_C[i] - expected[i]));
            naive_ok = 0; break;
        }
    }

    // ---- TILED ----
    matmul_tiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);   // warmup
    CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    CHECK_KERNEL();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float tiled_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    int tiled_ok = 1;
    for (int i = 0; i < M * N; ++i) {
        if (fabsf(h_C[i] - expected[i]) > EPS) {
            printf("tiled mismatch at %d: gpu=%f cpu=%f diff=%f\n",
                   i, h_C[i], expected[i], fabsf(h_C[i] - expected[i]));
            tiled_ok = 0; break;
        }
    }

    printf("Naive : %s   %.4f ms\n", naive_ok ? "PASS" : "FAIL", naive_ms);
    printf("Tiled : %s   %.4f ms\n", tiled_ok ? "PASS" : "FAIL", tiled_ms);
    if (tiled_ms > 0.0f)
        printf("Tiled speedup over naive: %.2fx\n", naive_ms / tiled_ms);

    free(expected);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
}