#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

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
void tensor_set_grad(Tensor *t, int *indices, float val);
void tensor_set_grad_scalar(Tensor *t, float val);
float tensor_get(Tensor *t, int *indices);
void tensor_set(Tensor *t, int *indices, float val);
void tensor_set_flat(Tensor *t, int index, float val);
void tensor_print(Tensor *t);
Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_mul(Tensor *a, Tensor *b);
Tensor *tensor_matmul(Tensor *a, Tensor *b);
Tensor *tensor_relu(Tensor *a);
Tensor *tensor_sigmoid(Tensor *a);
Tensor *tensor_log(Tensor *a);
Tensor *tensor_sum(Tensor *a);
void tensor_backward(Tensor *t);

#endif