#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "formula_eval.h"
#include "kof_parser.h"

static void compute_strides(const int *shape, int ndim, int *strides) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];
}

static size_t total_size(const int *shape, int ndim) {
    size_t t = 1;
    for (int i = 0; i < ndim; ++i) t *= (size_t)shape[i];
    return t;
}

void convn_forward_custom(
    float *output,
    int *out_shape,
    const float *input,
    const int *in_shape,
    int ndim,
    const int *kernel_offsets,
    const float *kernel_weights,
    int k_elems,
    const KOFKernel *kernel_def
){
    assert(ndim >= 1);
    assert(k_elems >= 1);

    FormulaSet *formula = formula_create(kernel_def);
    if (!formula) {
        fprintf(stderr, "Failed to compile formulas.\n");
        return;
    }

    // Determine kernel offset min/max for each dim
    int *min_off = (int*)malloc(sizeof(int)*ndim);
    int *max_off = (int*)malloc(sizeof(int)*ndim);
    for (int d = 0; d < ndim; ++d) {
        int mn = kernel_offsets[d];
        int mx = kernel_offsets[d];
        for (int t = 1; t < k_elems; ++t) {
            int off = kernel_offsets[t*ndim + d];
            if (off < mn) mn = off;
            if (off > mx) mx = off;
        }
        min_off[d] = mn;
        max_off[d] = mx;
    }

    // Compute output shape
    int *out_min = (int*)malloc(sizeof(int)*ndim);
    int *out_max = (int*)malloc(sizeof(int)*ndim);
    int valid = 1;
    for (int d = 0; d < ndim; ++d) {
        out_min[d] = -min_off[d];
        out_max[d] = in_shape[d] - 1 - max_off[d];
        int len = out_max[d] - out_min[d] + 1;
        if (len < 0) { len = 0; valid = 0; }
        out_shape[d] = len;
    }

    size_t out_total = total_size(out_shape, ndim);
    if (!valid || out_total == 0) {
        free(min_off); free(max_off); free(out_min); free(out_max);
        formula_free(formula);
        return;
    }

    int *in_strides  = (int*)malloc(sizeof(int)*ndim);
    int *out_strides = (int*)malloc(sizeof(int)*ndim);
    compute_strides(in_shape,  ndim, in_strides);
    compute_strides(out_shape, ndim, out_strides);

    int *idx = (int*)calloc(ndim, sizeof(int));
    for (size_t flat_out = 0; flat_out < out_total; ++flat_out) {
        float acc = 0.0f;

        for (int t = 0; t < k_elems; ++t) {
            int in_flat = 0;
            float tap[MAX_VARS] = {0};
            for (int d = 0; d < ndim; ++d) {
                int coord = idx[d] + out_min[d] + kernel_offsets[t*ndim + d];
                tap[d+1] = (float)kernel_offsets[t*ndim + d];
                in_flat += coord * in_strides[d];
            }

            float x = input[in_flat];

            // Choose input encoder
            ExprSet *input_enc = (formula->tap_input_encodes && formula->tap_input_encodes[t])
                ? formula->tap_input_encodes[t]
                : formula->input_encode;
            float x_enc = exprset_eval(input_enc, (float[]){ x });

            // Choose kernel encoder
            tap[0] = kernel_weights[t];
            ExprSet *kernel_enc = (formula->tap_kernel_encodes && formula->tap_kernel_encodes[t])
                ? formula->tap_kernel_encodes[t]
                : formula->kernel_encode;
            float w_enc = exprset_eval(kernel_enc, tap);

            // Choose operation
            ExprSet *op = (formula->tap_ops && formula->tap_ops[t])
                ? formula->tap_ops[t]
                : formula->op;
            float combined = exprset_eval(op, (float[]){ x_enc, w_enc });

            acc += combined;
        }
        output[flat_out] = acc;

        // Advance index
        for (int d = ndim - 1; d >= 0; --d) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    free(idx);
    free(in_strides);
    free(out_strides);
    free(min_off);
    free(max_off);
    free(out_min);
    free(out_max);
    formula_free(formula);
}

