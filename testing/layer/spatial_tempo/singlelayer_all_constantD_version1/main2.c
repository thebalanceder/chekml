#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void convn_forward(float *output, const float *input, const float *kernel,
                   const int *input_shape, const int *kernel_shape, int ndim);

void run_convn_test(int ndim) {
    int total_in = 1, total_k = 1;
    int input_shape[5], kernel_shape[5];

    // Example: Input shape = [3, 3, 3, 3, 3]
    for (int i = 0; i < ndim; ++i) {
        input_shape[i] = 3 + i;         // e.g., 3,4,5,...
        kernel_shape[i] = 2;            // kernel size 2 in each dim
        total_in *= input_shape[i];
        total_k *= kernel_shape[i];
    }

    int output_shape[5], total_out = 1;
    for (int i = 0; i < ndim; ++i) {
        output_shape[i] = input_shape[i] - kernel_shape[i] + 1;
        total_out *= output_shape[i];
    }

    float *input = malloc(sizeof(float) * total_in);
    float *kernel = malloc(sizeof(float) * total_k);
    float *output = calloc(total_out, sizeof(float));

    // Fill input and kernel with increasing values
    for (int i = 0; i < total_in; ++i) input[i] = i + 1;
    for (int i = 0; i < total_k; ++i) kernel[i] = 1.0f;  // Simple averaging kernel

    convn_forward(output, input, kernel, input_shape, kernel_shape, ndim);

    printf("\n--- %dD Convolution Output ---\n", ndim);
    for (int i = 0; i < total_out && i < 64; ++i) {  // Only show up to 64 elements
        printf("%.1f ", output[i]);
        if ((i+1) % 8 == 0) printf("\n");
    }
    printf("\n");

    free(input);
    free(kernel);
    free(output);
}

int main() {
    for (int ndim = 1; ndim <= 5; ++ndim)
        run_convn_test(ndim);
    return 0;
}

