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

typedef struct {
    int size;
    float *data;
} Tensor;

Tensor *tensor_create_stub(int size) {
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));

    if (!tensor) {
        fprintf(stderr, "Failed to allocate memory for Tensor\n");
        return NULL;
    }

    tensor->size = size;
    tensor->data = (float *)malloc(size * sizeof(float));

    if (!tensor->data) {
        fprintf(stderr, "Failed to allocate memory for Tensor data\n");
        free(tensor);
        return NULL;
    }

    return tensor;
}


int main() {
    Tensor *tensor = tensor_create_stub(10);
    if (!tensor) {
        return EXIT_FAILURE;
    }

    // Initialize the tensor data
    for (int i = 0; i < tensor->size; i++) {
        tensor->data[i] = (float)i;
    }

    size_t bytes = tensor->size * sizeof(float);
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, tensor->data, bytes, cudaMemcpyHostToDevice));

    float *roundtripped = (float *)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(roundtripped, d_data, bytes, cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < tensor->size; i++) {
        if (tensor->data[i] != roundtripped[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        printf("Roundtrip successful!\n");
    } else {
        printf("Roundtrip failed!\n");
    }

    cudaFree(d_data);
    free(roundtripped);
    free(tensor->data);
    free(tensor);

    return 0;
}