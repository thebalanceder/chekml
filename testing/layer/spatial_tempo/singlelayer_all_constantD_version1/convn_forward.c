#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void compute_strides(const int *shape, int ndim, int *strides) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// Recursively iterate over N-dimensional output space
void convn_forward_recursive(
    float *output, const float *input, const float *kernel,
    const int *input_shape, const int *kernel_shape, const int *output_shape,
    const int *input_strides, const int *kernel_strides, const int *output_strides,
    int ndim, int dim, int *output_index)
{
    if (dim == ndim) {
        float acc = 0.0f;

        for (int k = 0; k < 1; ++k) {
            // Inner loop over kernel
            int *kernel_index = malloc(ndim * sizeof(int));
            for (int i = 0; i < ndim; ++i) kernel_index[i] = 0;

            while (1) {
                // Compute input_index = output_index + kernel_index
                int input_flat_index = 0;
                int kernel_flat_index = 0;
                for (int i = 0; i < ndim; ++i) {
                    int in_idx = output_index[i] + kernel_index[i];
                    if (in_idx < 0 || in_idx >= input_shape[i]) goto skip;

                    input_flat_index += in_idx * input_strides[i];
                    kernel_flat_index += kernel_index[i] * kernel_strides[i];
                }

                acc += input[input_flat_index] * kernel[kernel_flat_index];

                // Next kernel index
                int d = ndim - 1;
                while (d >= 0) {
                    kernel_index[d]++;
                    if (kernel_index[d] < kernel_shape[d]) break;
                    kernel_index[d] = 0;
                    d--;
                }
                if (d < 0) break;
            }
        skip:
            free(kernel_index);
        }

        // Write output
        int output_flat_index = 0;
        for (int i = 0; i < ndim; ++i) {
            output_flat_index += output_index[i] * output_strides[i];
        }
        output[output_flat_index] = acc;
        return;
    }

    for (int i = 0; i < output_shape[dim]; ++i) {
        output_index[dim] = i;
        convn_forward_recursive(output, input, kernel,
            input_shape, kernel_shape, output_shape,
            input_strides, kernel_strides, output_strides,
            ndim, dim + 1, output_index);
    }
}

void convn_forward(
    float *output, const float *input, const float *kernel,
    const int *input_shape, const int *kernel_shape, int ndim)
{
    int *output_shape = malloc(sizeof(int) * ndim);
    for (int i = 0; i < ndim; ++i) {
        output_shape[i] = input_shape[i] - kernel_shape[i] + 1;
    }

    int *input_strides = malloc(sizeof(int) * ndim);
    int *kernel_strides = malloc(sizeof(int) * ndim);
    int *output_strides = malloc(sizeof(int) * ndim);
    compute_strides(input_shape, ndim, input_strides);
    compute_strides(kernel_shape, ndim, kernel_strides);
    compute_strides(output_shape, ndim, output_strides);

    int *output_index = calloc(ndim, sizeof(int));
    convn_forward_recursive(output, input, kernel,
        input_shape, kernel_shape, output_shape,
        input_strides, kernel_strides, output_strides,
        ndim, 0, output_index);

    free(output_shape);
    free(input_strides);
    free(kernel_strides);
    free(output_strides);
    free(output_index);
}

