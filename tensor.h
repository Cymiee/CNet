#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef TENSOR_H
#define TENSOR_H

typedef struct Tensor Tensor;
typedef struct BackwardContext BackwardContext;

struct Tensor{
    int ndim;
    int size;
    int requires_grad;
    int *shape;
    int *strides;
    float *data;
    float *grad;
    BackwardContext *ctx;
};

struct BackwardContext{
    int num_inputs;
    void *extra;
    Tensor **inputs;
    Tensor *output;
    void (*backward_fn)(BackwardContext *ctx, Tensor *grad_output);
};

Tensor *tensor_create(int ndim, int *shape, int requires_grad);
void tensor_free(Tensor *t);
float tensor_get(Tensor *t, int *indices);
void tensor_set(Tensor *t, int *indices, float val);
void tensor_print(Tensor *t);
Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_mul(Tensor *a, Tensor *b);

#endif