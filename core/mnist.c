#include "mnist.h"
#include <stdio.h>

int swap_endian(int val) {
    return ((val >> 24) & 0xFF) | 
           ((val << 8) & 0xFF0000) | 
           ((val >> 8) & 0xFF00) | 
           ((val << 24) & 0xFF000000);
}

MNISTData *mnist_load(const char *image_path, const char *label_path) {
    MNISTData *data = malloc(sizeof(MNISTData));
    if (data == NULL) {
        fprintf(stderr, "malloc failed\n");
        return NULL;
    }
    
    FILE *imagef = fopen(image_path, "rb");
    FILE *labelf = fopen(label_path, "rb");

    if (imagef == NULL || labelf == NULL) {
        fprintf(stderr, "Failed to open files\n");
        free(data);
        return NULL;
    }

    int magic, count, rows, cols;
    fread(&magic, 4, 1, imagef);
    magic = swap_endian(magic);
    fread(&count, 4, 1, imagef);
    count = swap_endian(count);
    fread(&rows, 4, 1, imagef);
    rows = swap_endian(rows);
    fread(&cols, 4, 1, imagef);
    cols = swap_endian(cols);

    int datatype = (magic & 0xFF00) >> 8;
    int ndims = magic & 0xFF;

    if (datatype != 0x08 || ndims != 3) {
        fprintf(stderr, "Invalid image file format\n");
        fclose(imagef);
        fclose(labelf);
        free(data);
        return NULL;
    }

    data->images = malloc(count * rows * cols * sizeof(float));
    if (data->images == NULL) {
        fprintf(stderr, "malloc failed\n");
        fclose(imagef);
        fclose(labelf);
        free(data);
        return NULL;
    }
    data->labels = malloc(count * sizeof(int));
    if (data->labels == NULL) {
        fprintf(stderr, "malloc failed\n");
        free(data->images);
        fclose(imagef);
        fclose(labelf);
        free(data);
        return NULL;
    }

    int total = count * rows * cols;
    unsigned char *buf = malloc(total);
    if (buf == NULL) {
        fprintf(stderr, "malloc failed\n");
        free(data->images);
        free(data->labels);
        fclose(imagef);
        fclose(labelf);
        free(data);
        return NULL;
    }

    fread(buf, 1, total, imagef);
    for (int i = 0; i < total; i++)
        data->images[i] = buf[i] / 255.0f;
    free(buf);

    int lmagic, lcount;
    fread(&lmagic, 4, 1, labelf);
    lmagic = swap_endian(lmagic);
    fread(&lcount, 4, 1, labelf);
    lcount = swap_endian(lcount);

    if (lmagic != 0x00000801 || lcount != count) {
        fprintf(stderr, "Invalid label file format\n");
        fclose(imagef);
        fclose(labelf);
        free(data->images);
        free(data->labels);
        free(data);
        return NULL;
    }

    unsigned char *lbuf = malloc(lcount); 
    if (lbuf == NULL) {
        fprintf(stderr, "malloc failed\n");
        free(data->images);
        free(data->labels);
        fclose(imagef);
        fclose(labelf);
        free(data);
        return NULL;        
    }

    fread(lbuf, 1, lcount, labelf);
    for (int i = 0; i < count; i++)
        data->labels[i] = lbuf[i];
    free(lbuf);

    fclose(imagef);
    fclose(labelf);

    data->count = count;
    data->rows = rows;
    data->cols = cols;

    return data;
}

void mnist_free(MNISTData *data) {
    if (data) {
        free(data->images);
        free(data->labels);
        free(data);
    }
}
