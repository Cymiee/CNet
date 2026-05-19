#include "tensor.h"

Tensor *tensor_create(int ndim, int *shape, int requires_grad){
    Tensor *t = malloc(sizeof(Tensor));
    if (t==NULL){
        fprintf(stderr, "malloc failed\n");
        return NULL;
    }
    
    int size = 1;
    for(int i = 0; i<ndim; i++){
        size *= shape[i];
    }
    t->ndim = ndim;
    t->size=size;
    t->requires_grad=requires_grad;
    t->ctx=NULL;

    t->data = calloc(size, sizeof(float));
    if(t->data==NULL){
        fprintf(stderr, "malloc failed\n");
        free(t);
        return NULL;
    }
    if (requires_grad == 1){
        t->grad = calloc(size, sizeof(float));
        if(t->grad==NULL){
            fprintf(stderr, "malloc failed\n");
            free(t->data);
            free(t);
            return NULL;
        }
    } else
        t->grad = NULL;

    t->shape = malloc(ndim * sizeof(int));
    if(t->shape == NULL){
        fprintf(stderr, "malloc failed\n");
        free(t->data);
        free(t->grad);
        free(t);
        return NULL;
    }
    memcpy(t->shape, shape, ndim*sizeof(int));

    t->strides = malloc(ndim * sizeof(int));
    if(t->strides == NULL){
        fprintf(stderr, "malloc failed\n");
        free(t->data);
        free(t->grad);
        free(t->shape);
        free(t);
        return NULL;
    }
    t->strides[ndim-1] = 1;
    for(int i = ndim-2; i>=0; i--){
        t->strides[i] = t->strides[i+1] * shape[i+1];
    }

    return t;
}

void tensor_free(Tensor *t){
    free(t->strides);
    free(t->data);
    free(t->grad);
    free(t->shape);
    free(t);
}

float tensor_get(Tensor *t, int *indices){
    int offset = 0;
    for(int i = 0; i<t->ndim; i++){
        if(indices[i] < 0 || indices[i]>=t->shape[i]){
            fprintf(stderr, "index %d out of bounds for dimension %d (size %d)\n", indices[i], i, t->shape[i]);
            return -1.0f;
        }
        offset += indices[i]*t->strides[i];
    }
    return t->data[offset];
}

void tensor_set(Tensor *t, int *indices, float val){
    int offset = 0;
    for(int i = 0; i<t->ndim; i++){
        if(indices[i] < 0 || indices[i]>=t->shape[i]){
            fprintf(stderr, "index %d out of bounds for dimension %d (size %d)\n", indices[i], i, t->shape[i]);
            return;
        }
        offset += indices[i]*t->strides[i];
    }
    t->data[offset]=val;
}

void print_array(float *arr, int len){
    printf("[");
    for(int i=0; i<len-1; i++){
        printf("%f, ", arr[i]);
    }
    printf("%f]", arr[len-1]);
}

static void print_recursive(Tensor *t, int dim, int offset, int depth){
    if(dim == t->ndim - 1){
        print_array(&t->data[offset], t->shape[dim]);
    } else {
        printf("[");
        for(int i = 0; i<t->shape[dim]; i++){
            if(i>0)
                for(int d=0; d<depth; d++) printf(" ");
            print_recursive(t, dim + 1, offset + i * t->strides[dim], depth+1);
            if(i < t->shape[dim]-1) printf(",\n");
        }
        printf("]");
    }
}

void tensor_print(Tensor *t){
    print_recursive(t, 0, 0, 0);
    printf("\n");
}

Tensor *tensor_add(Tensor *a, Tensor *b){

}