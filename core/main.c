#include <stdio.h>
#include <string.h>
#include "tensor.h"
#include "mnist.h"

float forward_sum_mul(Tensor *a, Tensor *b) {
    Tensor *xy = tensor_mul(a, b);
    Tensor *s  = tensor_sum(xy);
    float val  = s->data[0];
    tensor_free(s);
    tensor_free(xy);
    return val;
}

int grad_check_sum_mul(Tensor *x, Tensor *y, float epsilon, float tolerance) {
    Tensor *xy = tensor_mul(x, y);
    Tensor *loss = tensor_sum(xy);
    loss->grad[0] = 1.0f;
    tensor_backward(loss);
    tensor_free(loss);
    tensor_free(xy);

    int passed = 1;
    for (int i = 0; i < x->size; i++) {
        float orig = x->data[i];

        x->data[i] = orig + epsilon;
        float fplus = forward_sum_mul(x, y);

        x->data[i] = orig - epsilon;
        float fminus = forward_sum_mul(x, y);

        x->data[i] = orig;

        float numerical = (fplus - fminus) / (2.0f * epsilon);
        float analytical = x->grad[i];
        float diff = fabsf(numerical - analytical);
        float norm = fabsf(numerical) + fabsf(analytical) + 1e-8f;
        float error = diff / norm;

        if (error > tolerance) {
            printf("FAIL at index %d: numerical=%.6f analytical=%.6f error=%.6f\n", 
                i, numerical, analytical, error);
            passed = 0;
        }
    }
    if (passed) printf("PASSED: all gradients are within tolerance %.0e\n", tolerance);
    return passed;
}
float forward_sum_matmul(Tensor *a, Tensor *b) {
    Tensor *xy = tensor_matmul(a, b);
    Tensor *s  = tensor_sum(xy);
    float val  = s->data[0];
    tensor_free(s);
    tensor_free(xy);
    return val;
}

int grad_check_sum_matmul(Tensor *x, Tensor *y, float epsilon, float tolerance) {
    Tensor *xy = tensor_matmul(x, y);
    Tensor *loss = tensor_sum(xy);
    loss->grad[0] = 1.0f;
    tensor_backward(loss);
    tensor_free(loss);
    tensor_free(xy);

    int passed = 1;
    for (int i = 0; i < x->size; i++) {
        float orig = x->data[i];

        x->data[i] = orig + epsilon;
        float fplus = forward_sum_matmul(x, y);

        x->data[i] = orig - epsilon;
        float fminus = forward_sum_matmul(x, y);

        x->data[i] = orig;

        float numerical = (fplus - fminus) / (2.0f * epsilon);
        float analytical = x->grad[i];
        float diff = fabsf(numerical - analytical);
        float norm = fabsf(numerical) + fabsf(analytical) + 1e-8f;
        float error = diff / norm;

        if (error > tolerance) {
            printf("FAIL at index %d: numerical=%.6f analytical=%.6f error=%.6f\n", 
                i, numerical, analytical, error);
            passed = 0;
        }
    }
    if (passed) printf("PASSED: all gradients are within tolerance %.0e\n", tolerance);
    return passed;
}

void test_ops() {
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
    tensor_free(c); tensor_free(b); tensor_free(a);

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
    tensor_free(s3); tensor_free(s2); tensor_free(s1);

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
    tensor_free(m3); tensor_free(m2); tensor_free(m1);

    printf("\n=== Test 4: tensor_relu ===\n");
    int shape1d[] = {6};
    Tensor *r1 = tensor_create(1, shape1d, 0);
    r1->data[0]=-3.0f; r1->data[1]=-1.0f; r1->data[2]=0.0f;
    r1->data[3]=1.0f;  r1->data[4]=2.0f;  r1->data[5]=5.0f;
    Tensor *r2 = tensor_relu(r1);
    printf("expect [0, 0, 0, 1, 2, 5]:\n");
    tensor_print(r2);
    tensor_free(r2); tensor_free(r1);

    printf("\n=== Test 5: tensor_sigmoid ===\n");
    Tensor *sig1 = tensor_create(1, shape1d, 0);
    sig1->data[0]=0.0f; sig1->data[1]=2.0f; sig1->data[2]=-2.0f;
    sig1->data[3]=10.0f; sig1->data[4]=-10.0f; sig1->data[5]=1.0f;
    Tensor *sig2 = tensor_sigmoid(sig1);
    printf("expect [0.5, 0.8808, 0.1192, ~1.0, ~0.0, 0.7311]:\n");
    tensor_print(sig2);
    tensor_free(sig2); tensor_free(sig1);

    printf("\n=== Test 6: tensor_log ===\n");
    int shape_log[] = {4};
    Tensor *l1 = tensor_create(1, shape_log, 0);
    l1->data[0]=1.0f; l1->data[1]=2.718f;
    l1->data[2]=10.0f; l1->data[3]=0.5f;
    Tensor *l2 = tensor_log(l1);
    printf("expect [0.0, ~1.0, 2.3026, -0.6931]:\n");
    tensor_print(l2);
    tensor_free(l2); tensor_free(l1);

    printf("\n=== Test 7: tensor_matmul ===\n");
    int sha[] = {2, 2};
    int shb[] = {2, 3};
    Tensor *ma = tensor_create(2, sha, 0);
    Tensor *mb = tensor_create(2, shb, 0);
    ma->data[0]=1.0f; ma->data[1]=2.0f;
    ma->data[2]=3.0f; ma->data[3]=4.0f;
    mb->data[0]=1.0f; mb->data[1]=2.0f; mb->data[2]=3.0f;
    mb->data[3]=4.0f; mb->data[4]=5.0f; mb->data[5]=6.0f;
    Tensor *mc = tensor_matmul(ma, mb);
    printf("expect [[9,12,15],[19,26,33]]:\n");
    tensor_print(mc);
    tensor_free(mc); tensor_free(mb); tensor_free(ma);

    printf("\n=== Test 8: tensor_sum ===\n");
    Tensor *t1 = tensor_create(2, shape, 0);
    t1->data[0]=1.0f; t1->data[1]=2.0f; t1->data[2]=3.0f;
    t1->data[3]=4.0f; t1->data[4]=5.0f; t1->data[5]=6.0f;
    Tensor *t2 = tensor_sum(t1);
    printf("expect [21.0]:\n");
    tensor_print(t2);
    tensor_free(t2); tensor_free(t1);

    printf("\n=== Test 9: shape mismatches ===\n");
    int shape3[] = {3, 2};
    Tensor *e1 = tensor_create(2, shape, 0);
    Tensor *e2 = tensor_create(2, shape3, 0);
    Tensor *e3 = tensor_add(e1, e2);
    printf("expect NULL (add): %p\n", (void*)e3);
    int shx[] = {2, 3};
    int shy[] = {2, 2};
    Tensor *mx = tensor_create(2, shx, 0);
    Tensor *my = tensor_create(2, shy, 0);
    Tensor *mz = tensor_matmul(mx, my);
    printf("expect NULL (matmul): %p\n", (void*)mz);
    tensor_free(e2); tensor_free(e1);
    tensor_free(mx); tensor_free(my);
}

void test_autograd() {
    // --- Test 1: simple multiply ---
    printf("=== Autograd Test 1: z = x * y ===\n");
    int shape1[] = {1};
    Tensor *x = tensor_create(1, shape1, 1);
    Tensor *y = tensor_create(1, shape1, 1);
    x->data[0] = 3.0f;
    y->data[0] = 4.0f;
    Tensor *z = tensor_mul(x, y);
    z->grad[0] = 1.0f;
    tensor_backward(z);
    printf("dz/dx expect 4.0: %f\n", x->grad[0]);
    printf("dz/dy expect 3.0: %f\n", y->grad[0]);
    tensor_free(z); tensor_free(y); tensor_free(x);

    // --- Test 2: chained ops z = relu(a + b) * c ---
    printf("\n=== Autograd Test 2: z = relu(a + b) * c ===\n");
    // a=2, b=-1, c=3
    // a+b = 1, relu(1) = 1, z = 1*3 = 3
    // dz/dc = relu(a+b) = 1
    // dz/d(relu) = c = 3
    // relu backward: input was 1 > 0 so passes through → dz/d(a+b) = 3
    // add backward: dz/da = 3, dz/db = 3
    Tensor *ta = tensor_create(1, shape1, 1);
    Tensor *tb = tensor_create(1, shape1, 1);
    Tensor *tc = tensor_create(1, shape1, 1);
    ta->data[0] = 2.0f;
    tb->data[0] = -1.0f;
    tc->data[0] = 3.0f;
    Tensor *tab  = tensor_add(ta, tb);
    Tensor *trel = tensor_relu(tab);
    Tensor *tz   = tensor_mul(trel, tc);
    tz->grad[0] = 1.0f;
    tensor_backward(tz);
    printf("dz/da expect 3.0: %f\n", ta->grad[0]);
    printf("dz/db expect 3.0: %f\n", tb->grad[0]);
    printf("dz/dc expect 1.0: %f\n", tc->grad[0]);
    tensor_free(tz); tensor_free(trel); tensor_free(tab);
    tensor_free(tc); tensor_free(tb); tensor_free(ta);

    // --- Test 3: z = sum(x * y) with 2D tensors ---
    printf("\n=== Autograd Test 3: z = sum(x * y), 2D tensors ===\n");
    // x = [[1,2],[3,4]], y = [[2,2],[2,2]]
    // x*y = [[2,4],[6,8]], sum = 20
    // dz/dx[i] = y[i], dz/dy[i] = x[i]
    int shape2d[] = {2, 2};
    Tensor *px = tensor_create(2, shape2d, 1);
    Tensor *py = tensor_create(2, shape2d, 1);
    px->data[0]=1.0f; px->data[1]=2.0f;
    px->data[2]=3.0f; px->data[3]=4.0f;
    py->data[0]=2.0f; py->data[1]=2.0f;
    py->data[2]=2.0f; py->data[3]=2.0f;

    printf("Grad check: \n");
    grad_check_sum_mul(px, py, 1e-4f, 1e-3f);

    printf("dz/dx expect [[2,2],[2,2]]:\n");
    Tensor *gx = tensor_create(2, shape2d, 0);
    for (int i = 0; i < 4; i++) gx->data[i] = px->grad[i];
    tensor_print(gx);
    printf("dz/dy expect [[1,2],[3,4]]:\n");
    Tensor *gy = tensor_create(2, shape2d, 0);
    for (int i = 0; i < 4; i++) gy->data[i] = py->grad[i];
    tensor_print(gy);
    tensor_free(gx); tensor_free(gy);
    tensor_free(py); tensor_free(px);

    // --- Test 4: z = sum(matmul(A, B)) ---
    printf("\n=== Autograd Test 4: z = sum(matmul(A, B)) ===\n");
    // A = [[1,2],[3,4]], B = [[1,0],[0,1]] (identity)
    // matmul(A,B) = A, sum = 10
    // dz/dA = ones * B^T = ones (since B is identity)
    // dz/dB = A^T * ones
    int shA[] = {2, 2};
    int shB[] = {2, 2};
    Tensor *A = tensor_create(2, shA, 1);
    Tensor *B = tensor_create(2, shB, 1);
    A->data[0]=1.0f; A->data[1]=2.0f;
    A->data[2]=3.0f; A->data[3]=4.0f;
    B->data[0]=1.0f; B->data[1]=0.0f;
    B->data[2]=0.0f; B->data[3]=1.0f;

    printf("Grad check: \n");
    grad_check_sum_matmul(A, B, 1e-4f, 1e-3f);
    printf("dz/dA expect [[1,1],[1,1]]:\n");
    Tensor *gA = tensor_create(2, shA, 0);
    for (int i = 0; i < 4; i++) gA->data[i] = A->grad[i];
    tensor_print(gA);
    printf("dz/dB expect [[4,4],[6,6]] (A^T * ones... wait: col sums of A^T):\n");
    printf("dz/dB expect [[1+3, 1+3],[2+4, 2+4]] = [[4,4],[6,6]]:\n");
    Tensor *gB = tensor_create(2, shB, 0);
    for (int i = 0; i < 4; i++) gB->data[i] = B->grad[i];
    tensor_print(gB);
    tensor_free(gA); tensor_free(gB);
    tensor_free(B); tensor_free(A);
}

void test_mnist() {
    printf("=== MNIST Test: Loading ===\n");

    MNISTData *train = mnist_load(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte"
    );

    printf("count: %d\n", train->count);
    printf("rows: %d, cols: %d\n", train->rows, train->cols);
    printf("first label: %d\n", train->labels[0]);
    printf("first pixel: %f\n", train->images[0]);

    mnist_free(train);
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Usage: ./tensor [ops|autograd|mnist|all]\n");
        return 0;
    }
    if (strcmp(argv[1], "ops") == 0) {
        test_ops();
    } else if (strcmp(argv[1], "autograd") == 0) {
        test_autograd();
    } else if (strcmp(argv[1], "mnist") == 0) {
        test_mnist();
    } else if (strcmp(argv[1], "all") == 0) {
        test_ops();
        printf("\n");
        test_autograd();
        printf("\n");
        test_mnist();
    } else {
        printf("Unknown command: %s\n", argv[1]);
        printf("Usage: ./tensor [ops|autograd|all]\n");
    }
    return 0;
}