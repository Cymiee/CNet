#ifndef MNIST_H
#define MNIST_H

#include <stdlib.h>

typedef struct {
    float *images;
    int *labels;
    int count;
    int rows;
    int cols;
} MNISTData;

MNISTData *mnist_load(const char *image_path, const char *label_path);
void mnist_free(MNISTData *data);

#endif