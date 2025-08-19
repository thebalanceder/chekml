// main.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Declaration
void convn_forward_custom(
    float *output,
    int *out_shape,               // OUT
    const float *input,
    const int *in_shape,
    int ndim,
    const int *kernel_offsets,
    const float *kernel_weights,
    int k_elems
);

// Helper to print a 2D tensor
static void print_2d(const float *a, int h, int w, const char *title) {
    printf("%s (%dx%d):\n", title, h, w);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            printf("%7.2f ", a[i*w + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    // Example: 2D input 6x6 with values 1..36
    int ndim = 2;
    int in_shape[2] = {6, 6};
    float *input = (float*)malloc(sizeof(float) * in_shape[0]*in_shape[1]);
    for (int i = 0; i < in_shape[0]*in_shape[1]; ++i) input[i] = (float)(i+1);

    // ----- Kernel A: irregular "plus" shape (center + 4 arms) -----
    // Offsets are (dy, dx): center (0,0), up (-1,0), down(+1,0), left(0,-1), right(0,+1)
    // Weights can be whatever you like; here we just use 1 for all taps.
    int plus_k_elems = 5;
    int plus_offsets[] = {
        0, 0,
       -1, 0,
        1, 0,
        0,-1,
        0, 1
    };
    float plus_weights[] = {
        1, 1, 1, 1, 1
    };

    int out_shape_plus[2] = {0, 0};
    // Output size upper bound: safe to allocate full input size (or compute exact out_shape first)
    float *out_plus = (float*)calloc(in_shape[0]*in_shape[1], sizeof(float));

    convn_forward_custom(
        out_plus, out_shape_plus,
        input, in_shape, ndim,
        plus_offsets, plus_weights, plus_k_elems
    );

    // ----- Kernel B: circle-like kernel within radius R (sampled) -----
    // Build offsets inside circle of radius 1.5 around (0,0), weight = 1 (or Gaussian if you prefer)
    int cap = 64;
    int *circ_offsets = (int*)malloc(sizeof(int)*cap*2);
    float *circ_weights = (float*)malloc(sizeof(float)*cap);
    int circ_k = 0;
    float R = 1.5f;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            float r = sqrtf((float)(dy*dy + dx*dx));
            if (r <= R) {
                if (circ_k >= cap) { fprintf(stderr, "increase cap\n"); exit(1); }
                circ_offsets[circ_k*2 + 0] = dy;
                circ_offsets[circ_k*2 + 1] = dx;
                // e.g., uniform weights, or Gaussian: exp(-r^2 / (2*sigma^2))
                circ_weights[circ_k] = 1.0f;
                circ_k++;
            }
        }
    }

    int out_shape_circ[2] = {0, 0};
    float *out_circ = (float*)calloc(in_shape[0]*in_shape[1], sizeof(float));
    convn_forward_custom(
        out_circ, out_shape_circ,
        input, in_shape, ndim,
        circ_offsets, circ_weights, circ_k
    );

    // Print results
    print_2d(input, in_shape[0], in_shape[1], "Input");

    print_2d(out_plus, out_shape_plus[0], out_shape_plus[1], "Output (Plus kernel)");

    print_2d(out_circ, out_shape_circ[0], out_shape_circ[1], "Output (Circle-like kernel)");

    // Cleanup
    free(input);
    free(out_plus);
    free(out_circ);
    free(circ_offsets);
    free(circ_weights);
    return 0;
}

