#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kof_parser.h"
#include "formula_eval.h"
#include "temporal_wrapper.h"

// Forward & backward from your earlier files
void convn_forward_custom(
    float *output, int *out_shape,
    const float *input, const int *in_shape, int ndim,
    const int *kernel_offsets, const float *kernel_weights, int k_elems,
    const KOFKernel *K);

void convn_backward_custom(
    const float *input, const int *in_shape, int ndim,
    const int *offsets, const float *weights, int k_elems,
    const float *d_output, const int *out_shape,
    float *d_input, float *d_weights,
    const KOFKernel *K);

static size_t total_size(const int *shape, int ndim) {
    size_t t = 1;
    for (int i = 0; i < ndim; ++i) t *= (size_t)shape[i];
    return t;
}

// Stable demo input
static void fill_ramp(float *a, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = (float)(i % 17) / 16.0f;
}

// MSE with correct gradient scaling for mean loss: dL/dy = (y - t) / N
static float mse_and_grad(const float *y, const float *t, float *dLdy, size_t n) {
    float loss = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float e = y[i] - t[i];
        loss += 0.5f * e * e;
        dLdy[i] = e;
    }
    loss /= (float)n;
    float invn = 1.0f / (float)n;
    for (size_t i = 0; i < n; ++i) dLdy[i] *= invn;
    return loss;
}

// SGD with optional weight decay and gradient clipping
static void sgd_update(float *w, const float *g, int n, float lr, float wd, float clip) {
    for (int i = 0; i < n; ++i) {
        float gi = g[i];
        if (clip > 0.0f) {
            if (gi >  clip) gi =  clip;
            if (gi < -clip) gi = -clip;
        }
        w[i] -= lr * (gi + wd * w[i]);
    }
}

static void print_shape(const char *name, const int *s, int n) {
    printf("%s=[", name);
    for (int i = 0; i < n; ++i) { printf("%d%s", s[i], (i+1<n)?",":""); }
    printf("]\n");
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s kernel1.kof kernel2.kof temporal1.tprl temporal2.tprl [epochs=30] [lr=0.005] [weight_decay=1e-4] [clip=1.0]\n", argv[0]);
        return 1;
    }
    int   E    = (argc >= 6) ? atoi(argv[5])   : 30;
    float lr   = (argc >= 7) ? (float)atof(argv[6]) : 0.005f;
    float wd   = (argc >= 8) ? (float)atof(argv[7]) : 1e-4f;
    float clip = (argc >= 9) ? (float)atof(argv[8]) : 1.0f;

    // Load two kernels
    KOFKernel K1 = {0}, K2 = {0};
    if (kof_load(argv[1], &K1) != 0) {
        fprintf(stderr, "Failed to load KOF: %s\n", argv[1]);
        return 2;
    }
    if (kof_load(argv[2], &K2) != 0) {
        fprintf(stderr, "Failed to load KOF: %s\n", argv[2]);
        kof_free(&K1);
        return 2;
    }
    if (K1.ndim != 3 || K2.ndim != 3) {
        fprintf(stderr, "This trainer expects ndim=3 in both KOF files (got %d, %d)\n", K1.ndim, K2.ndim);
        kof_free(&K1); kof_free(&K2);
        return 3;
    }
    printf("Loaded KOF1: ndim=%d, taps=%d, norm=%d, scale=%.3f\n", K1.ndim, K1.k_elems, (int)K1.norm, K1.scale);
    printf("Loaded KOF2: ndim=%d, taps=%d, norm=%d, scale=%.3f\n", K2.ndim, K2.k_elems, (int)K2.norm, K2.scale);

    // Create temporal wrappers
    TemporalWrapper *W1 = temporal_wrapper_create(&K1, argv[3]);
    TemporalWrapper *W2 = temporal_wrapper_create(&K2, argv[4]);
    if (!W1 || !W2) {
        fprintf(stderr, "Failed to create temporal wrappers\n");
        kof_free(&K1); kof_free(&K2);
        temporal_wrapper_free(W1); temporal_wrapper_free(W2);
        return 5;
    }

    int ndim = 3;
    int in_shape[3] = {6, 6, 6};
    size_t in_total = total_size(in_shape, ndim);
    float *input = (float*)malloc(sizeof(float) * in_total);
    if (!input) {
        fprintf(stderr, "Failed to allocate input memory\n");
        kof_free(&K1); kof_free(&K2); temporal_wrapper_free(W1); temporal_wrapper_free(W2);
        return 6;
    }
    fill_ramp(input, in_total);

    // First convolution
    int out_shape1[3] = {0, 0, 0};
    float *output1 = (float*)calloc(in_total, sizeof(float));
    if (!output1) {
        fprintf(stderr, "Failed to allocate output1 memory\n");
        free(input); kof_free(&K1); kof_free(&K2); temporal_wrapper_free(W1); temporal_wrapper_free(W2);
        return 6;
    }
    temporal_wrapper_forward(W1, output1, out_shape1, input, in_shape, ndim);
    print_shape("in_shape", in_shape, 3);
    print_shape("out_shape1", out_shape1, 3);
    size_t out_total1 = total_size(out_shape1, ndim);
    if (out_total1 == 0) {
        fprintf(stderr, "No valid output region for first kernel and input size.\n");
        free(input); free(output1); kof_free(&K1); kof_free(&K2); temporal_wrapper_free(W1); temporal_wrapper_free(W2);
        return 4;
    }

    // Second convolution
    int out_shape2[3] = {0, 0, 0};
    float *output2 = (float*)calloc(out_total1, sizeof(float));
    if (!output2) {
        fprintf(stderr, "Failed to allocate output2 memory\n");
        free(input); free(output1); kof_free(&K1); kof_free(&K2); temporal_wrapper_free(W1); temporal_wrapper_free(W2);
        return 6;
    }
    temporal_wrapper_forward(W2, output2, out_shape2, output1, out_shape1, ndim);
    print_shape("out_shape2", out_shape2, 3);
    size_t out_total2 = total_size(out_shape2, ndim);
    if (out_total2 == 0) {
        fprintf(stderr, "No valid output region for second kernel and first layer output size.\n");
        free(input); free(output1); free(output2); kof_free(&K1); kof_free(&K2); temporal_wrapper_free(W1); temporal_wrapper_free(W2);
        return 4;
    }

    float *target = (float*)malloc(sizeof(float) * out_total2);
    if (!target) {
        fprintf(stderr, "Failed to allocate target memory\n");
        free(input); free(output1); free(output2); kof_free(&K1); kof_free(&K2); temporal_wrapper_free(W1); temporal_wrapper_free(W2);
        return 6;
    }
    for (size_t i = 0; i < out_total2; ++i) target[i] = 0.25f;

    float *dLdy = (float*)malloc(sizeof(float) * out_total2);
    float *d_output1 = (float*)calloc(out_total1, sizeof(float));
    float *d_input = (float*)calloc(in_total, sizeof(float));
    float *d_weight1 = (float*)calloc(K1.k_elems, sizeof(float));
    float *d_weight2 = (float*)calloc(K2.k_elems, sizeof(float));
    if (!dLdy || !d_output1 || !d_input || !d_weight1 || !d_weight2) {
        fprintf(stderr, "Failed to allocate gradient memory\n");
        free(input); free(output1); free(output2); free(target); free(dLdy); free(d_output1); free(d_input); free(d_weight1); free(d_weight2);
        kof_free(&K1); kof_free(&K2); temporal_wrapper_free(W1); temporal_wrapper_free(W2);
        return 6;
    }

    for (int epoch = 1; epoch <= E; ++epoch) {
        // Forward pass: Layer 1
        temporal_wrapper_forward(W1, output1, out_shape1, input, in_shape, ndim);
        // Forward pass: Layer 2
        temporal_wrapper_forward(W2, output2, out_shape2, output1, out_shape1, ndim);

        // Compute loss
        float loss = mse_and_grad(output2, target, dLdy, out_total2);

        // Backward pass: Layer 2
        memset(d_output1, 0, sizeof(float) * out_total1);
        memset(d_weight2, 0, sizeof(float) * K2.k_elems);
        temporal_wrapper_backward(W2, output1, out_shape1, ndim, dLdy, out_shape2, d_output1, d_weight2);

        // Backward pass: Layer 1
        memset(d_input, 0, sizeof(float) * in_total);
        memset(d_weight1, 0, sizeof(float) * K1.k_elems);
        temporal_wrapper_backward(W1, input, in_shape, ndim, d_output1, out_shape1, d_input, d_weight1);

        // Update weights and k parameters
        sgd_update(K1.weights, d_weight1, K1.k_elems, lr, wd, clip);
        sgd_update(K2.weights, d_weight2, K2.k_elems, lr, wd, clip);
        sgd_update(&W1->k, &W1->k_grad, 1, lr, wd, clip);
        sgd_update(&W2->k, &W2->k_grad, 1, lr, wd, clip);

        // Update wrappers with gradients and loss
        temporal_wrapper_update(W1, d_weight1, loss, output1, out_shape1, input, in_shape, ndim);
        temporal_wrapper_update(W2, d_weight2, loss, output2, out_shape2, output1, out_shape1, ndim);

        if (epoch == 1 || epoch % 5 == 0 || epoch == E) {
            printf("epoch %3d: loss=%.6f, k1=%.6f, k2=%.6f\n", epoch, loss, W1->k, W2->k);
        }
    }

    printf("First 8 weights after training (Layer 1):\n");
    for (int i = 0; i < K1.k_elems && i < 8; ++i) {
        printf("  w[%d]=%.6f\n", i, K1.weights[i]);
    }
    printf("First 8 weights after training (Layer 2):\n");
    for (int i = 0; i < K2.k_elems && i < 8; ++i) {
        printf("  w[%d]=%.6f\n", i, K2.weights[i]);
    }

    free(input);
    free(output1);
    free(output2);
    free(target);
    free(dLdy);
    free(d_output1);
    free(d_input);
    free(d_weight1);
    free(d_weight2);
    kof_free(&K1);
    kof_free(&K2);
    temporal_wrapper_free(W1);
    temporal_wrapper_free(W2);
    return 0;
}
