#include "tensor.h"

int main(){
    // --- Test 1: basic addition ---
    printf("=== Test 1: basic addition ===\n");
    int shape[] = {2, 3};
    Tensor *a = tensor_create(2, shape, 1);
    Tensor *b = tensor_create(2, shape, 0);

    // a = [[1, 2, 3], [4, 5, 6]]
    a->data[0]=1.0f; a->data[1]=2.0f; a->data[2]=3.0f;
    a->data[3]=4.0f; a->data[4]=5.0f; a->data[5]=6.0f;

    // b = [[10, 20, 30], [40, 50, 60]]
    b->data[0]=10.0f; b->data[1]=20.0f; b->data[2]=30.0f;
    b->data[3]=40.0f; b->data[4]=50.0f; b->data[5]=60.0f;

    Tensor *c = tensor_add(a, b);
    printf("expect [[11, 22, 33], [44, 55, 66]]:\n");
    tensor_print(c);

    // c should require grad since a does
    printf("expect requires_grad=1: %d\n", c->requires_grad);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);

    // --- Test 2: both require grad ---
    printf("\n=== Test 2: requires_grad propagation ===\n");
    Tensor *x = tensor_create(2, shape, 0);
    Tensor *y = tensor_create(2, shape, 1);
    Tensor *z = tensor_add(x, y);
    printf("expect requires_grad=1: %d\n", z->requires_grad);
    tensor_free(x);
    tensor_free(y);
    tensor_free(z);

    // --- Test 3: neither requires grad ---
    printf("\n=== Test 3: no grad propagation ===\n");
    Tensor *p = tensor_create(2, shape, 0);
    Tensor *q = tensor_create(2, shape, 0);
    Tensor *r = tensor_add(p, q);
    printf("expect requires_grad=0: %d\n", r->requires_grad);
    tensor_free(p);
    tensor_free(q);
    tensor_free(r);

    // --- Test 4: shape mismatch ---
    printf("\n=== Test 4: shape mismatch ===\n");
    int shape2[] = {3, 2};
    Tensor *m = tensor_create(2, shape, 0);
    Tensor *n = tensor_create(2, shape2, 0);
    Tensor *o = tensor_add(m, n);
    printf("expect NULL: %p\n", (void*)o);
    tensor_free(m);
    tensor_free(n);
    // do NOT free o, it's NULL

    return 0;
}