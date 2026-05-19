#include "tensor.h"

int main(){
    int shape[] = {2, 3};
    Tensor *t = tensor_create(2, shape, 1);

    // set some values
    int idx00[] = {0, 0}; tensor_set(t, idx00, 1.0f);
    int idx01[] = {0, 1}; tensor_set(t, idx01, 2.0f);
    int idx10[] = {1, 0}; tensor_set(t, idx10, 3.0f);
    int idx11[] = {1, 2}; tensor_set(t, idx11, 9.0f);

    // get them back
    printf("expect 1.0: %f\n", tensor_get(t, idx00));
    printf("expect 2.0: %f\n", tensor_get(t, idx01));
    printf("expect 3.0: %f\n", tensor_get(t, idx10));
    printf("expect 9.0: %f\n", tensor_get(t, idx11));

    tensor_free(t);
    printf("\n\n");

    int shape3d[] = {2, 3, 4};

    Tensor *t3 = tensor_create(3, shape3d, 0);
    for(int i = 0; i < t3->size; i++)
        t3->data[i] = (float)i + 1.0f;
    tensor_print(t3);
    tensor_free(t3);

    return 0;
}