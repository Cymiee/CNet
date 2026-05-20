#include "tensor.h"

Tensor *tensor_create(int ndim, int *shape, int requires_grad) {
    Tensor *t = malloc(sizeof(Tensor));
    if (t == NULL) {
        fprintf(stderr, "malloc failed\n");
        return NULL;
    }

    int size = 1;
    for (int i = 0; i < ndim; i++)
        size *= shape[i];

    t->ndim          = ndim;
    t->size          = size;
    t->requires_grad = requires_grad;
    t->ctx           = NULL;

    t->data = calloc(size, sizeof(float));
    if (t->data == NULL) {
        fprintf(stderr, "malloc failed\n");
        free(t);
        return NULL;
    }

    if (requires_grad) {
        t->grad = calloc(size, sizeof(float));
        if (t->grad == NULL) {
            fprintf(stderr, "malloc failed\n");
            free(t->data);
            free(t);
            return NULL;
        }
    } else {
        t->grad = NULL;
    }

    t->shape = malloc(ndim * sizeof(int));
    if (t->shape == NULL) {
        fprintf(stderr, "malloc failed\n");
        free(t->data);
        free(t->grad);
        free(t);
        return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(int));

    t->strides = malloc(ndim * sizeof(int));
    if (t->strides == NULL) {
        fprintf(stderr, "malloc failed\n");
        free(t->data);
        free(t->grad);
        free(t->shape);
        free(t);
        return NULL;
    }
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        t->strides[i] = t->strides[i + 1] * shape[i + 1];

    return t;
}

static BackwardContext *backward_context_create(int num_inputs, Tensor *output, void (*backward_fn)(BackwardContext*, Tensor*)) {
    BackwardContext *ctx = malloc(sizeof(BackwardContext));
    if (ctx == NULL) {
        fprintf(stderr, "malloc failed\n");
        return NULL;
    }
    ctx->inputs = malloc(num_inputs * sizeof(Tensor *));
    if(ctx->inputs == NULL) {
        fprintf(stderr, "malloc failed\n");
        free(ctx);
        return NULL;
    }
    ctx->num_inputs = num_inputs;
    ctx->output = output;
    ctx->extra = NULL;
    ctx->backward_fn = backward_fn;
    return ctx;
}

void tensor_free(Tensor *t) {
    free(t->strides);
    free(t->data);
    free(t->grad);
    free(t->shape);
    free(t);
}

static void backward_context_free(BackwardContext *ctx) {
    free(ctx->inputs);
    if (ctx->extra != NULL) free(ctx->extra);
    free(ctx);
}

float tensor_get(Tensor *t, int *indices) {
    int offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (indices[i] < 0 || indices[i] >= t->shape[i]) {
            fprintf(stderr, "index %d out of bounds for dimension %d (size %d)\n",
                    indices[i], i, t->shape[i]);
            return -1.0f;
        }
        offset += indices[i] * t->strides[i];
    }
    return t->data[offset];
}

void tensor_set(Tensor *t, int *indices, float val) {
    int offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (indices[i] < 0 || indices[i] >= t->shape[i]) {
            fprintf(stderr, "index %d out of bounds for dimension %d (size %d)\n",
                    indices[i], i, t->shape[i]);
            return;
        }
        offset += indices[i] * t->strides[i];
    }
    t->data[offset] = val;
}

static void print_array(float *arr, int len) {
    printf("[");
    for (int i = 0; i < len - 1; i++)
        printf("%f, ", arr[i]);
    printf("%f]", arr[len - 1]);
}

static void print_recursive(Tensor *t, int dim, int offset, int depth) {
    if (dim == t->ndim - 1) {
        print_array(&t->data[offset], t->shape[dim]);
    } else {
        printf("[");
        for (int i = 0; i < t->shape[dim]; i++) {
            if (i > 0)
                for (int d = 0; d < depth; d++) printf(" ");
            print_recursive(t, dim + 1, offset + i * t->strides[dim], depth + 1);
            if (i < t->shape[dim] - 1) printf(",\n");
        }
        printf("]");
    }
}

void tensor_print(Tensor *t) {
    print_recursive(t, 0, 0, 0);
    printf("\n");
}

static int tensor_shape_equal(Tensor *a, Tensor *b) {
    if (a->ndim != b->ndim)
        return 0;
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i])
            return 0;
    return 1;
}

static void add_backward(BackwardContext *ctx, Tensor *grad_output) {
    if(ctx->inputs[0]->requires_grad)
        for (int i = 0; i < ctx->inputs[0]->size; i++)
            ctx->inputs[0]->grad[i] += grad_output->data[i];
    if(ctx->inputs[1]->requires_grad)
        for (int i = 0; i < ctx->inputs[1]->size; i++)
            ctx->inputs[1]->grad[i] += grad_output->data[i];
}

static void sub_backward(BackwardContext *ctx, Tensor *grad_output) {
    if(ctx->inputs[0]->requires_grad)
        for (int i = 0; i < ctx->inputs[0]->size; i++)
            ctx->inputs[0]->grad[i] += grad_output->data[i];
    if(ctx->inputs[1]->requires_grad)
        for (int i = 0; i < ctx->inputs[1]->size; i++)
            ctx->inputs[1]->grad[i] -= grad_output->data[i];
}

static void mul_backward(BackwardContext *ctx, Tensor *grad_output) {
    if(ctx->inputs[0]->requires_grad)
        for (int i = 0; i < ctx->inputs[0]->size; i++)
            ctx->inputs[0]->grad[i] += grad_output->data[i] * ctx->inputs[1]->data[i];
    if(ctx->inputs[1]->requires_grad)
        for (int i = 0; i < ctx->inputs[1]->size; i++)
            ctx->inputs[1]->grad[i] += grad_output->data[i] * ctx->inputs[0]->data[i];
}

static void relu_backward(BackwardContext *ctx, Tensor *grad_output) {
    if(ctx->inputs[0]->requires_grad)
        for (int i = 0; i < ctx->inputs[0]->size; i++)
            ctx->inputs[0]->grad[i] += grad_output->data[i] * (ctx->inputs[0]->data[i] > 0 ? 1 : 0);
}

static void sigmoid_backward(BackwardContext *ctx, Tensor *grad_output) {
    if(ctx->inputs[0]->requires_grad)
        for (int i = 0; i < ctx->inputs[0]->size; i++)
            ctx->inputs[0]->grad[i] += grad_output->data[i] * ctx->output->data[i] * (1 - ctx->output->data[i]);
}

static void log_backward(BackwardContext *ctx, Tensor *grad_output) {
    if(ctx->inputs[0]->requires_grad)
        for (int i = 0; i < ctx->inputs[0]->size; i++)
            ctx->inputs[0]->grad[i] += grad_output->data[i] * (1.0f / ctx->inputs[0]->data[i]);
}

static void sum_backward(BackwardContext *ctx, Tensor *grad_output) {
    if(ctx->inputs[0]->requires_grad)
        for (int i = 0; i < ctx->inputs[0]->size; i++)
            ctx->inputs[0]->grad[i] += grad_output->data[0];
}

static void matmul_backward(BackwardContext *ctx, Tensor *grad_output) {
    int rows = ctx->inputs[0]->shape[0];
    int sdim = ctx->inputs[0]->shape[1];
    int cols = ctx->inputs[1]->shape[1];

    if (ctx->inputs[0]->requires_grad)
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < sdim; k++)
                for (int j = 0; j < cols; j++)
                    ctx->inputs[0]->grad[i * sdim + k] +=
                        grad_output->data[i * cols + j] *
                        ctx->inputs[1]->data[k * cols + j];
    if (ctx->inputs[1]->requires_grad)
        for (int k = 0; k < sdim; k++)
            for (int j = 0; j < cols; j++)
                for (int i = 0; i < rows; i++)                
                    ctx->inputs[1]->grad[k * cols + j] +=
                        ctx->inputs[0]->data[i * sdim + k] *
                        grad_output->data[i * cols + j];
}

Tensor *tensor_add(Tensor *a, Tensor *b) {
    if (!tensor_shape_equal(a, b)) {
        fprintf(stderr, "tensor shapes not equal, cannot add\n");
        return NULL;
    }
    int dim = a->ndim, *shape = a->shape, size = a->size;
    Tensor *c = tensor_create(dim, shape, a->requires_grad || b->requires_grad);
    for (int i = 0; i < size; i++)
        c->data[i] = a->data[i] + b->data[i];
    
    if (a->requires_grad || b->requires_grad){
        BackwardContext *ctx = backward_context_create(2, c, add_backward);
        if (ctx == NULL) { tensor_free(c); return NULL; }
        ctx->inputs[0] = a;
        ctx->inputs[1] = b;
        c->ctx = ctx;
    }

    return c;
}

Tensor *tensor_sub(Tensor *a, Tensor *b) {
    if (!tensor_shape_equal(a, b)) {
        fprintf(stderr, "tensor shapes not equal, cannot subtract\n");
        return NULL;
    }
    int dim = a->ndim, *shape = a->shape, size = a->size;
    Tensor *c = tensor_create(dim, shape, a->requires_grad || b->requires_grad);
    for (int i = 0; i < size; i++)
        c->data[i] = a->data[i] - b->data[i];

    if (a->requires_grad || b->requires_grad){
        BackwardContext *ctx = backward_context_create(2, c, sub_backward);
        if (ctx == NULL) { tensor_free(c); return NULL; }
        ctx->inputs[0] = a;
        ctx->inputs[1] = b;
        c->ctx = ctx;
    }

    return c;
}

Tensor *tensor_mul(Tensor *a, Tensor *b) {
    if (!tensor_shape_equal(a, b)) {
        fprintf(stderr, "tensor shapes not equal, cannot multiply\n");
        return NULL;
    }
    int dim = a->ndim, *shape = a->shape, size = a->size;
    Tensor *c = tensor_create(dim, shape, a->requires_grad || b->requires_grad);
    for (int i = 0; i < size; i++)
        c->data[i] = a->data[i] * b->data[i];
    
    if (a->requires_grad || b->requires_grad){
        BackwardContext *ctx = backward_context_create(2, c, mul_backward);
        if (ctx == NULL) { tensor_free(c); return NULL; }
        ctx->inputs[0] = a;
        ctx->inputs[1] = b;
        c->ctx = ctx;
    }

    return c;
}

Tensor *tensor_relu(Tensor *a) {
    int dim = a->ndim, *shape = a->shape, size = a->size;
    Tensor *b = tensor_create(dim, shape, a->requires_grad);
    for (int i = 0; i < size; i++)
        b->data[i] = a->data[i] > 0 ? a->data[i] : 0.0f;

    if (a->requires_grad){
        BackwardContext *ctx = backward_context_create(1, b, relu_backward);
        if (ctx == NULL) { tensor_free(b); return NULL; }
        ctx->inputs[0] = a;
        b->ctx = ctx;
    }

    return b;
}

Tensor *tensor_sigmoid(Tensor *a) {
    int dim = a->ndim, *shape = a->shape, size = a->size;
    Tensor *b = tensor_create(dim, shape, a->requires_grad);
    for (int i = 0; i < size; i++)
        b->data[i] = 1.0f / (1.0f + expf(-a->data[i]));

    if (a->requires_grad){
        BackwardContext *ctx = backward_context_create(1, b, sigmoid_backward);
        if (ctx == NULL) { tensor_free(b); return NULL; }
        ctx->inputs[0] = a;
        b->ctx = ctx;
    }

    return b;
}

Tensor *tensor_log(Tensor *a) {
    int dim = a->ndim, *shape = a->shape, size = a->size;
    Tensor *b = tensor_create(dim, shape, a->requires_grad);
    for (int i = 0; i < size; i++)
        b->data[i] = logf(a->data[i]);

    if (a->requires_grad){
        BackwardContext *ctx = backward_context_create(1, b, log_backward);
        if (ctx == NULL) { tensor_free(b); return NULL; }
        ctx->inputs[0] = a;
        b->ctx = ctx;
    }

    return b;
}

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "matmul failed, both tensors have to be 2-D\n");
        return NULL;
    }
    if (a->shape[1] != b->shape[0]) {
        fprintf(stderr, "matmul failed, tensor a cols must equal tensor b rows\n");
        return NULL;
    }
    int rows = a->shape[0], cols = b->shape[1], sdim = a->shape[1];
    int shape[2] = {rows, cols};
    Tensor *c = tensor_create(2, shape, a->requires_grad || b->requires_grad);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float val = 0.0f;
            for (int k = 0; k < sdim; k++)
                val += a->data[i * sdim + k] * b->data[k * cols + j];
            c->data[i * cols + j] = val;
        }
    }

    if (a->requires_grad || b->requires_grad){
        BackwardContext *ctx = backward_context_create(2, c, matmul_backward);
        if (ctx == NULL) { tensor_free(c); return NULL; }
        ctx->inputs[0] = a;
        ctx->inputs[1] = b;
        c->ctx = ctx;
    }
    
    return c;
}

Tensor *tensor_sum(Tensor *a) {
    int shape[1] = {1};
    Tensor *b = tensor_create(1, shape, a->requires_grad);
    for (int i = 0; i < a->size; i++)
        b->data[0] += a->data[i];

    if (a->requires_grad){
        BackwardContext *ctx = backward_context_create(1, b, sum_backward);
        if (ctx == NULL) { tensor_free(b); return NULL; }
        ctx->inputs[0] = a;
        b->ctx = ctx;
    }

    return b;
}