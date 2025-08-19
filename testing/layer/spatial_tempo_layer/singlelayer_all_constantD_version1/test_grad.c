// test_grad.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

// Forward/backward from your files
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

static size_t total_size(const int *shape, int ndim) {
    size_t t = 1;
    for (int i = 0; i < ndim; ++i) t *= (size_t)shape[i];
    return t;
}

// Utility: compute L = sum(output) given current input & weights.
// Also returns out_shape and (optionally) stores output if outbuf != NULL
static float loss_sum_forward(
    float *outbuf, int *out_shape,
    const float *input, const int *in_shape, int ndim,
    const int *offsets, const float *weights, int k_elems
){
    // allocate a temp buffer big enough; forward writes only valid region (prod(out_shape))
    int max_out = 1;
    for (int i = 0; i < ndim; ++i) max_out *= in_shape[i]; // upper bound
    float *tmp = outbuf ? outbuf : (float*)calloc(max_out, sizeof(float));

    convn_forward_custom(tmp, out_shape, input, in_shape, ndim, offsets, weights, k_elems);

    size_t out_total = total_size(out_shape, ndim);
    float L = 0.0f;
    for (size_t i = 0; i < out_total; ++i) L += tmp[i];

    if (!outbuf) free(tmp);
    return L;
}

static float randf() {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; // [-1,1]
}

static void fill_random(float *a, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = randf();
}

static void compare_arrays(const char *name,
                           const float *a, const float *b, size_t n,
                           float *max_abs_err_out, float *max_rel_err_out)
{
    float max_abs = 0.0f, max_rel = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float aa = a[i], bb = b[i];
        float abs_err = fabsf(aa - bb);
        float denom = fmaxf(1e-6f, fabsf(aa) + fabsf(bb));
        float rel_err = abs_err / denom;
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;
    }
    if (max_abs_err_out) *max_abs_err_out = max_abs;
    if (max_rel_err_out) *max_rel_err_out = max_rel;
    printf("%s: max_abs_err=%.3e, max_rel_err=%.3e\n", name, max_abs, max_rel);
}

int main(void) {
    srand(12345);

    // --- Small 2D test (change to 1D/3D/ND as you like) ---
    int ndim = 2;
    int in_shape[2] = {5, 4};             // small input
    size_t in_total = total_size(in_shape, ndim);

    // Irregular kernel example: the 2D “plus”
    int k_elems = 5;
    int offsets[] = {
        0,  0,
       -1,  0,
        1,  0,
        0, -1,
        0,  1
    };
    float *weights = (float*)malloc(sizeof(float)*k_elems);
    for (int t = 0; t < k_elems; ++t) weights[t] = randf();

    float *input = (float*)malloc(sizeof(float)*in_total);
    fill_random(input, in_total);

    // Forward to get out_shape
    int out_shape[2] = {0,0};
    float L0 = loss_sum_forward(NULL, out_shape, input, in_shape, ndim, offsets, weights, k_elems);
    size_t out_total = total_size(out_shape, ndim);
    printf("out_shape = [%d, %d], out_total=%zu\n", out_shape[0], out_shape[1], out_total);

    // Analytical grads: d_out = 1 (since L=sum(output))
    float *d_out = (float*)malloc(sizeof(float)*out_total);
    for (size_t i = 0; i < out_total; ++i) d_out[i] = 1.0f;
    float *d_input = (float*)calloc(in_total, sizeof(float));
    float *d_weights = (float*)calloc(k_elems, sizeof(float));

    convn_backward_custom(input, in_shape, ndim,
                          offsets, weights, k_elems,
                          d_out, out_shape,
                          d_input, d_weights);

    // --- Finite difference for input ---
    float eps = 1e-3f;
    float *fd_dinput = (float*)malloc(sizeof(float)*in_total);

    for (size_t i = 0; i < in_total; ++i) {
        float old = input[i];
        input[i] = old + eps;
        float Lp = loss_sum_forward(NULL, out_shape, input, in_shape, ndim, offsets, weights, k_elems);
        input[i] = old - eps;
        float Lm = loss_sum_forward(NULL, out_shape, input, in_shape, ndim, offsets, weights, k_elems);
        input[i] = old;
        fd_dinput[i] = (Lp - Lm) / (2.0f * eps);
    }

    // --- Finite difference for weights ---
    float *fd_dweights = (float*)malloc(sizeof(float)*k_elems);
    for (int t = 0; t < k_elems; ++t) {
        float old = weights[t];
        weights[t] = old + eps;
        float Lp = loss_sum_forward(NULL, out_shape, input, in_shape, ndim, offsets, weights, k_elems);
        weights[t] = old - eps;
        float Lm = loss_sum_forward(NULL, out_shape, input, in_shape, ndim, offsets, weights, k_elems);
        weights[t] = old;
        fd_dweights[t] = (Lp - Lm) / (2.0f * eps);
    }

    // Compare
    float max_abs, max_rel;
    compare_arrays("grad_input", d_input, fd_dinput, in_total, &max_abs, &max_rel);
    compare_arrays("grad_weights", d_weights, fd_dweights, k_elems, &max_abs, &max_rel);

    // Tight-ish tolerances (central diff, sum loss)
    if (max_abs < 5e-4f && max_rel < 5e-3f) {
        printf("✅ Gradient check PASSED\n");
    } else {
        printf("❌ Gradient check FAILED (tighten eps or inspect)\n");
    }

    free(weights);
    free(input);
    free(d_out);
    free(d_input);
    free(d_weights);
    free(fd_dinput);
    free(fd_dweights);
    return 0;
}

