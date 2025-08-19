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

void convn_backward_custom(
    const float *input,
    const int   *in_shape,
    int          ndim,
    const int   *offsets,
    const float *weights,
    int          k_elems,
    const float *d_output,
    const int   *out_shape,
    float       *d_input,
    float       *d_weights,
    const KOFKernel *kernel_def
){
    assert(ndim >= 1 && k_elems >= 1);

    FormulaSet *formula = formula_create(kernel_def);
    if (!formula) return;

    int *min_off = (int*)malloc(sizeof(int)*ndim);
    int *max_off = (int*)malloc(sizeof(int)*ndim);
    for (int d = 0; d < ndim; ++d) {
        int mn = offsets[d];
        int mx = offsets[d];
        for (int t = 1; t < k_elems; ++t) {
            int off = offsets[t*ndim + d];
            if (off < mn) mn = off;
            if (off > mx) mx = off;
        }
        min_off[d] = mn;
        max_off[d] = mx;
    }

    int *out_min = (int*)malloc(sizeof(int)*ndim);
    for (int d = 0; d < ndim; ++d)
        out_min[d] = -min_off[d];

    int *in_strides  = (int*)malloc(sizeof(int)*ndim);
    int *out_strides = (int*)malloc(sizeof(int)*ndim);
    compute_strides(in_shape,  ndim, in_strides);
    compute_strides(out_shape, ndim, out_strides);

    const size_t out_total = total_size(out_shape, ndim);

    int *idx = (int*)calloc(ndim, sizeof(int));
    for (size_t flat_out = 0; flat_out < out_total; ++flat_out) {
        float g = d_output[flat_out];

        for (int t = 0; t < k_elems; ++t) {
            int in_flat = 0;
            float tap[MAX_VARS] = {0};
            for (int d = 0; d < ndim; ++d) {
                int coord = idx[d] + out_min[d] + offsets[t*ndim + d];
                tap[d+1] = (float)offsets[t*ndim + d];
                in_flat += coord * in_strides[d];
            }

            float x = input[in_flat];

            // Select per-tap or global encoder
            ExprSet *input_enc = (formula->tap_input_encodes && formula->tap_input_encodes[t])
                ? formula->tap_input_encodes[t]
                : formula->input_encode;
            float x_enc = exprset_eval(input_enc, (float[]){ x });

            tap[0] = weights[t];
            ExprSet *kernel_enc = (formula->tap_kernel_encodes && formula->tap_kernel_encodes[t])
                ? formula->tap_kernel_encodes[t]
                : formula->kernel_encode;
            float w_enc = exprset_eval(kernel_enc, tap);

            ExprSet *op = (formula->tap_ops && formula->tap_ops[t])
                ? formula->tap_ops[t]
                : formula->op;
            float partial = exprset_eval(op, (float[]){ x_enc, w_enc });

            d_weights[t] += g * partial;       // ∂L/∂w
            d_input[in_flat] += g * w_enc;     // ∂L/∂x assuming d(f(x,w))/dx = w_enc (assumption: symmetric op)
        }

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
    formula_free(formula);
}
