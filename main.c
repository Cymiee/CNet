#include <stdio.h>
#include "tensor.h"

int main(){
    // --- Test 1: tensor_add ---
    printf("=== Test 1: tensor_add ===\n");
    int shape[] = {2, 3};
    Tensor *a = tensor_create(2, shape, 1);
    Tensor *b = tensor_create(2, shape, 0);
    a->data[0]=1.0f; a->data[1]=2.0f; a->data[2]=3.0f;
    a->data[3]=4.0f; a->data[4]=5.0f; a->data[5]=6.0f;
    b->data[0]=10.0f; b->data[1]=20.0f; b->data[2]=30.0f;
    b->data[3]=40.0f; b->data[4]=50.0f; b->data[5]=60.0f;
    Tensor *c = tensor_add(a, b);
    printf("expect [[11, 22, 33], [44, 55, 66]]:\n");
    tensor_print(c);
    printf("expect requires_grad=1: %d\n", c->requires_grad);
    tensor_free(a); tensor_free(b); tensor_free(c);

    // --- Test 2: tensor_sub ---
    printf("\n=== Test 2: tensor_sub ===\n");
    Tensor *s1 = tensor_create(2, shape, 0);
    Tensor *s2 = tensor_create(2, shape, 0);
    s1->data[0]=5.0f; s1->data[1]=6.0f; s1->data[2]=7.0f;
    s1->data[3]=8.0f; s1->data[4]=9.0f; s1->data[5]=10.0f;
    s2->data[0]=1.0f; s2->data[1]=2.0f; s2->data[2]=3.0f;
    s2->data[3]=4.0f; s2->data[4]=5.0f; s2->data[5]=6.0f;
    Tensor *s3 = tensor_sub(s1, s2);
    printf("expect [[4, 4, 4], [4, 4, 4]]:\n");
    tensor_print(s3);
    tensor_free(s1); tensor_free(s2); tensor_free(s3);

    // --- Test 3: tensor_mul ---
    printf("\n=== Test 3: tensor_mul ===\n");
    Tensor *m1 = tensor_create(2, shape, 0);
    Tensor *m2 = tensor_create(2, shape, 0);
    m1->data[0]=1.0f; m1->data[1]=2.0f; m1->data[2]=3.0f;
    m1->data[3]=4.0f; m1->data[4]=5.0f; m1->data[5]=6.0f;
    m2->data[0]=2.0f; m2->data[1]=2.0f; m2->data[2]=2.0f;
    m2->data[3]=2.0f; m2->data[4]=2.0f; m2->data[5]=2.0f;
    Tensor *m3 = tensor_mul(m1, m2);
    printf("expect [[2, 4, 6], [8, 10, 12]]:\n");
    tensor_print(m3);
    tensor_free(m1); tensor_free(m2); tensor_free(m3);

    // --- Test 4: tensor_relu ---
    printf("\n=== Test 4: tensor_relu ===\n");
    int shape1d[] = {6};
    Tensor *r1 = tensor_create(1, shape1d, 0);
    r1->data[0]=-3.0f; r1->data[1]=-1.0f; r1->data[2]=0.0f;
    r1->data[3]=1.0f;  r1->data[4]=2.0f;  r1->data[5]=5.0f;
    Tensor *r2 = tensor_relu(r1);
    printf("expect [0, 0, 0, 1, 2, 5]:\n");
    tensor_print(r2);
    tensor_free(r1); tensor_free(r2);

    // --- Test 5: tensor_sigmoid ---
    printf("\n=== Test 5: tensor_sigmoid ===\n");
    Tensor *sig1 = tensor_create(1, shape1d, 0);
    sig1->data[0]=0.0f;  // sigmoid(0) = 0.5
    sig1->data[1]=2.0f;  // sigmoid(2) ≈ 0.8808
    sig1->data[2]=-2.0f; // sigmoid(-2) ≈ 0.1192
    sig1->data[3]=10.0f; // sigmoid(10) ≈ 1.0
    sig1->data[4]=-10.0f;// sigmoid(-10) ≈ 0.0
    sig1->data[5]=1.0f;  // sigmoid(1) ≈ 0.7311
    Tensor *sig2 = tensor_sigmoid(sig1);
    printf("expect [0.5, 0.8808, 0.1192, ~1.0, ~0.0, 0.7311]:\n");
    tensor_print(sig2);
    tensor_free(sig1); tensor_free(sig2);

    // --- Test 6: tensor_log ---
    printf("\n=== Test 6: tensor_log ===\n");
    int shape_log[] = {4};
    Tensor *l1 = tensor_create(1, shape_log, 0);
    l1->data[0]=1.0f;    // log(1) = 0
    l1->data[1]=2.718f;  // log(e) ≈ 1.0
    l1->data[2]=10.0f;   // log(10) ≈ 2.3026
    l1->data[3]=0.5f;    // log(0.5) ≈ -0.6931
    Tensor *l2 = tensor_log(l1);
    printf("expect [0.0, ~1.0, 2.3026, -0.6931]:\n");
    tensor_print(l2);
    tensor_free(l1); tensor_free(l2);

    // --- Test 7: tensor_matmul ---
    printf("\n=== Test 7: tensor_matmul ===\n");
    int sha[] = {2, 2};
    int shb[] = {2, 3};
    Tensor *ma = tensor_create(2, sha, 0);
    Tensor *mb = tensor_create(2, shb, 0);
    // A = [[1,2],[3,4]]
    ma->data[0]=1.0f; ma->data[1]=2.0f;
    ma->data[2]=3.0f; ma->data[3]=4.0f;
    // B = [[1,2,3],[4,5,6]]
    mb->data[0]=1.0f; mb->data[1]=2.0f; mb->data[2]=3.0f;
    mb->data[3]=4.0f; mb->data[4]=5.0f; mb->data[5]=6.0f;
    Tensor *mc = tensor_matmul(ma, mb);
    printf("expect [[9,12,15],[19,26,33]]:\n");
    tensor_print(mc);
    tensor_free(ma); tensor_free(mb); tensor_free(mc);

    // --- Test 8: tensor_matmul shape mismatch ---
    printf("\n=== Test 8: tensor_matmul mismatch ===\n");
    int shx[] = {2, 3};
    int shy[] = {2, 2};
    Tensor *mx = tensor_create(2, shx, 0);
    Tensor *my = tensor_create(2, shy, 0);
    Tensor *mz = tensor_matmul(mx, my);
    printf("expect NULL: %p\n", (void*)mz);
    tensor_free(mx); tensor_free(my);

    // --- Test 9: tensor_sum ---
    printf("\n=== Test 9: tensor_sum ===\n");
    Tensor *t1 = tensor_create(2, shape, 0);
    t1->data[0]=1.0f; t1->data[1]=2.0f; t1->data[2]=3.0f;
    t1->data[3]=4.0f; t1->data[4]=5.0f; t1->data[5]=6.0f;
    Tensor *t2 = tensor_sum(t1);
    printf("expect [21.0]:\n");
    tensor_print(t2);
    tensor_free(t1); tensor_free(t2);

    // --- Test 10: shape mismatch on elementwise ops ---
    printf("\n=== Test 10: elementwise shape mismatch ===\n");
    int shape3[] = {3, 2};
    Tensor *e1 = tensor_create(2, shape, 0);
    Tensor *e2 = tensor_create(2, shape3, 0);
    Tensor *e3 = tensor_add(e1, e2);
    printf("expect NULL: %p\n", (void*)e3);
    tensor_free(e1); tensor_free(e2);

    return 0;
}