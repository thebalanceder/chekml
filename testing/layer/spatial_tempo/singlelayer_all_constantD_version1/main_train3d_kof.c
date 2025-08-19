// main_train3d_kof.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kof_parser.h"
#include "formula_eval.h"

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

// stable demo input
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
        // L2 regularization (weight decay)
        w[i] -= lr * (gi + wd * w[i]);
    }
}

static void print_shape(const char *name, const int *s, int n) {
    printf("%s=[", name);
    for (int i = 0; i < n; ++i) { printf("%d%s", s[i], (i+1<n)?",":""); }
    printf("]\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s kernel.kof [epochs=30] [lr=0.005] [weight_decay=1e-4] [clip=1.0]\n", argv[0]);
        return 1;
    }
    int   E    = (argc >= 3) ? atoi(argv[2])   : 30;
    float lr   = (argc >= 4) ? (float)atof(argv[3]) : 0.005f;
    float wd   = (argc >= 5) ? (float)atof(argv[4]) : 1e-4f;
    float clip = (argc >= 6) ? (float)atof(argv[5]) : 1.0f;

    KOFKernel K = {0};
    if (kof_load(argv[1], &K) != 0) {
        fprintf(stderr, "Failed to load KOF: %s\n", argv[1]);
        return 2;
    }
    if (K.ndim != 3) {
        fprintf(stderr, "This trainer expects ndim=3 in KOF (got %d)\n", K.ndim);
        kof_free(&K);
        return 3;
    }
    printf("Loaded KOF: ndim=%d, taps=%d, norm=%d, scale=%.3f\n", K.ndim, K.k_elems, (int)K.norm, K.scale);

    int ndim = 3;
    int in_shape[3] = {6, 6, 6};
    size_t in_total = total_size(in_shape, ndim);
    float *input = (float*)malloc(sizeof(float) * in_total);
    fill_ramp(input, in_total);

    int out_shape[3] = {0,0,0};
    float *output = (float*)calloc(in_total, sizeof(float));
    convn_forward_custom(output, out_shape, input, in_shape, ndim,
                         K.offsets, K.weights, K.k_elems, &K);

    print_shape("in_shape", in_shape, 3);
    print_shape("out_shape", out_shape, 3);
    size_t out_total = total_size(out_shape, ndim);
    if (out_total == 0) {
        fprintf(stderr, "No valid output region for this kernel and input size.\n");
        free(input); free(output); kof_free(&K);
        return 4;
    }

    float *target = (float*)malloc(sizeof(float) * out_total);
    for (size_t i = 0; i < out_total; ++i) target[i] = 0.25f;

    float *dLdy     = (float*)malloc(sizeof(float) * out_total);
    float *d_input  = (float*)calloc(in_total, sizeof(float));
    float *d_weight = (float*)calloc(K.k_elems, sizeof(float));

    for (int epoch = 1; epoch <= E; ++epoch) {
        convn_forward_custom(output, out_shape, input, in_shape, ndim,
                             K.offsets, K.weights, K.k_elems, &K);

        float loss = mse_and_grad(output, target, dLdy, out_total);

        memset(d_input,  0, sizeof(float)*in_total);
        memset(d_weight, 0, sizeof(float)*K.k_elems);
        convn_backward_custom(input, in_shape, ndim,
                              K.offsets, K.weights, K.k_elems,
                              dLdy, out_shape,
                              d_input, d_weight, &K);

        sgd_update(K.weights, d_weight, K.k_elems, lr, wd, clip);

        if (epoch == 1 || epoch % 5 == 0 || epoch == E) {
            printf("epoch %3d: loss=%.6f\n", epoch, loss);
        }
    }

    printf("First 8 weights after training:\n");
    for (int i = 0; i < K.k_elems && i < 8; ++i) {
        printf("  w[%d]=%.6f\n", i, K.weights[i]);
    }

    free(input);
    free(output);
    free(target);
    free(dLdy);
    free(d_input);
    free(d_weight);
    kof_free(&K);
    return 0;
}

