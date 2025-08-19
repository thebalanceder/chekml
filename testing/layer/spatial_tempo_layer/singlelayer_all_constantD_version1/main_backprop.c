// main_backprop.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void convn_forward_custom(
    float *output, int *out_shape,
    const float *input, const int *in_shape, int ndim,
    const int *kernel_offsets, const float *kernel_weights, int k_elems
);

void convn_backward_custom(
    const float *input, const int *in_shape, int ndim,
    const int *offsets, const float *weights, int k_elems,
    const float *d_output, const int *out_shape,
    float *d_input, float *d_weights
);

static void print_2d(const float *a, int h, int w, const char *title) {
    printf("%s (%dx%d):\n", title, h, w);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) printf("%7.2f ", a[i*w + j]);
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    // Input 2D: 5x5
    int ndim = 2;
    int in_shape[2] = {5, 5};
    int in_total = in_shape[0]*in_shape[1];

    float *input = (float*)malloc(sizeof(float)*in_total);
    for (int i = 0; i < in_total; ++i) input[i] = (float)(i+1);

    // "plus" kernel (5 taps) in 2D: (dy,dx)
    int k_elems = 5;
    int offsets[] = {
        0,  0,
       -1,  0,
        1,  0,
        0, -1,
        0,  1
    };
    float weights[] = {1, 1, 1, 1, 1};

    // Forward (get out_shape + output)
    int out_shape[2] = {0,0};
    float *output = (float*)calloc(in_total, sizeof(float)); // upper bound alloc
    convn_forward_custom(output, out_shape, input, in_shape, ndim, offsets, weights, k_elems);

    // Upstream gradient: set all ones over the valid region (same shape as output)
    int out_total = out_shape[0]*out_shape[1];
    float *d_out = (float*)malloc(sizeof(float)*out_total);
    for (int i = 0; i < out_total; ++i) d_out[i] = 1.0f;

    // Backward
    float *d_input   = (float*)calloc(in_total, sizeof(float)); // must be zero-initialized
    float *d_weights = (float*)calloc(k_elems,  sizeof(float)); // must be zero-initialized

    convn_backward_custom(
        input, in_shape, ndim,
        offsets, weights, k_elems,
        d_out, out_shape,
        d_input, d_weights
    );

    // Print results
    print_2d(input, in_shape[0], out_shape[1] ? in_shape[1] : 0, "Input");
    print_2d(output, out_shape[0], out_shape[1], "Forward Output");
    print_2d(d_input, in_shape[0], in_shape[1], "dInput");
    printf("dWeights: ");
    for (int t = 0; t < k_elems; ++t) printf("%.2f ", d_weights[t]);
    printf("\n");

    free(input);
    free(output);
    free(d_out);
    free(d_input);
    free(d_weights);
    return 0;
}

