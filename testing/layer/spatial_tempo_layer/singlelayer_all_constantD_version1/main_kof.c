// main_kof.c
#include <stdio.h>
#include <stdlib.h>
#include "kof_parser.h"

// from your previous file
void convn_forward_custom(
    float *output, int *out_shape,
    const float *input, const int *in_shape, int ndim,
    const int *kernel_offsets, const float *kernel_weights, int k_elems
);

static void print_2d(const float *a, int h, int w, const char *title) {
    printf("%s (%dx%d):\n", title, h, w);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) printf("%7.2f ", a[i*w + j]);
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv){
    if(argc < 2){
        fprintf(stderr, "Usage: %s <kernel.kof>\n", argv[0]);
        return 1;
    }

    // Example 2D input 6x6
    int ndim = 2;
    int in_shape[2] = {6, 6};
    float *input = (float*)malloc(sizeof(float)*in_shape[0]*in_shape[1]);
    for (int i = 0; i < in_shape[0]*in_shape[1]; ++i) input[i] = (float)(i+1);

    KOFKernel K = {0};
    if(kof_load(argv[1], &K) != 0){
        fprintf(stderr, "Failed to load kernel: %s\n", argv[1]);
        free(input);
        return 2;
    }
    if(K.ndim != ndim){
        fprintf(stderr, "Kernel ndim (%d) != input ndim (%d)\n", K.ndim, ndim);
        kof_free(&K); free(input); return 3;
    }

    int out_shape[2] = {0,0};
    float *output = (float*)calloc(in_shape[0]*in_shape[1], sizeof(float));

    convn_forward_custom(
        output, out_shape,
        input, in_shape, ndim,
        K.offsets, K.weights, K.k_elems
    );

    printf("Loaded kernel with %d taps, ndim=%d\n", K.k_elems, K.ndim);
    print_2d(input, in_shape[0], in_shape[1], "Input");
    print_2d(output, out_shape[0], out_shape[1], "Output");

    kof_free(&K);
    free(input);
    free(output);
    return 0;
}

