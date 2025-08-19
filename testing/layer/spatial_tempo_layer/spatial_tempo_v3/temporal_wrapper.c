#define _GNU_SOURCE
#include "temporal_wrapper.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "tinyexpr.h"

extern void convn_forward_custom(float *output, int *out_shape, const float *input, const int *in_shape, int ndim, const int *kernel_offsets, const float *kernel_weights, int k_elems, const KOFKernel *K);
extern void convn_backward_custom(const float *input, const int *in_shape, int ndim, const int *offsets, const float *weights, int k_elems, const float *d_output, const int *out_shape, float *d_input, float *d_weights, const KOFKernel *K);

static size_t total_size(const int *shape, int ndim) {
    size_t t = 1;
    for (int i = 0; i < ndim; ++i) t *= (size_t)shape[i];
    return t;
}

TemporalWrapper *temporal_wrapper_create(KOFKernel *kernel, const char *tprl_filename) {
    TemporalWrapper *w = (TemporalWrapper*)calloc(1, sizeof(TemporalWrapper));
    if (!w) return NULL;
    w->kernel = kernel;
    w->k_elems = kernel->k_elems;

    // Load temporal formula
    w->t_formula = (TemporalFormula*)calloc(1, sizeof(TemporalFormula));
    if (!w->t_formula || temporal_load(tprl_filename, w->t_formula) != 0) {
        fprintf(stderr, "Failed to load .tprl file: %s\n", tprl_filename);
        temporal_wrapper_free(w);
        return NULL;
    }

    w->weight_history = (float*)calloc(w->k_elems * w->t_formula->history_length, sizeof(float));
    w->gradient_history = (float*)calloc(w->k_elems * w->t_formula->history_length, sizeof(float));
    w->grad_magnitudes = (float*)calloc(w->k_elems, sizeof(float));
    w->selected_indices = (int*)calloc(w->k_elems, sizeof(int));
    if (!w->weight_history || !w->gradient_history || !w->grad_magnitudes || !w->selected_indices) {
        temporal_wrapper_free(w);
        return NULL;
    }
    w->history_count = 0;
    w->k = w->t_formula->k_init;
    w->k_grad = 0.0f;

    // Compile formula
    w->formula_expr = exprset_compile(w->t_formula->formula, (const char**)w->t_formula->vars, w->t_formula->var_count);
    if (!w->formula_expr) {
        fprintf(stderr, "Failed to compile temporal formula: %s\n", w->t_formula->formula);
        temporal_wrapper_free(w);
        return NULL;
    }
    return w;
}

void temporal_wrapper_free(TemporalWrapper *w) {
    if (!w) return;
    free(w->weight_history);
    free(w->gradient_history);
    free(w->grad_magnitudes);
    free(w->selected_indices);
    temporal_free(w->t_formula);
    free(w->t_formula);
    exprset_free(w->formula_expr);
    free(w);
}

static float compute_loss(const float *output, const float *target, size_t n) {
    float loss = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float e = output[i] - target[i];
        loss += 0.5f * e * e;
    }
    return loss / (float)n;
}

void temporal_wrapper_update(TemporalWrapper *w, const float *gradients, float loss, const float *output, const int *out_shape, const float *input, const int *in_shape, int ndim) {
    // Update history
    int max_history = w->t_formula->history_length;
    if (w->history_count < max_history) {
        memcpy(w->weight_history + w->k_elems * w->history_count, w->kernel->weights, w->k_elems * sizeof(float));
        memcpy(w->gradient_history + w->k_elems * w->history_count, gradients, w->k_elems * sizeof(float));
        w->history_count++;
    } else {
        memmove(w->weight_history, w->weight_history + w->k_elems, w->k_elems * (max_history - 1) * sizeof(float));
        memmove(w->gradient_history, w->gradient_history + w->k_elems, w->k_elems * (max_history - 1) * sizeof(float));
        memcpy(w->weight_history + w->k_elems * (max_history - 1), w->kernel->weights, w->k_elems * sizeof(float));
        memcpy(w->gradient_history + w->k_elems * (max_history - 1), gradients, w->k_elems * sizeof(float));
    }

    // Compute average gradient magnitudes
    for (int i = 0; i < w->k_elems; ++i) {
        w->grad_magnitudes[i] = 0.0f;
        for (int h = 0; h < w->history_count; ++h) {
            w->grad_magnitudes[i] += fabsf(w->gradient_history[h * w->k_elems + i]);
        }
        w->grad_magnitudes[i] /= (float)(w->history_count ? w->history_count : 1);
    }

    // Select weights with top 50% gradient magnitudes
    float max_grad = 0.0f;
    for (int i = 0; i < w->k_elems; ++i) {
        if (w->grad_magnitudes[i] > max_grad) max_grad = w->grad_magnitudes[i];
    }
    w->selected_count = 0;
    for (int i = 0; i < w->k_elems; ++i) {
        if (w->grad_magnitudes[i] >= 0.5f * max_grad) {
            w->selected_indices[w->selected_count++] = i;
        }
    }

    // Ablation testing: Skip if too few weights
    if (w->k_elems < 2 || w->selected_count == 0) return;

    size_t out_total = total_size(out_shape, ndim);
    float *temp_output = (float*)calloc(out_total, sizeof(float));
    int *temp_shape = (int*)calloc(ndim, sizeof(int));
    if (!temp_output || !temp_shape) {
        free(temp_output); free(temp_shape);
        return;
    }
    memcpy(temp_shape, out_shape, ndim * sizeof(int));
    float base_loss = loss;

    for (int i = 0; i < w->selected_count; ++i) {
        int idx = w->selected_indices[i];
        float orig_weight = w->kernel->weights[idx];
        w->kernel->weights[idx] = 0.0f; // Ablate
        convn_forward_custom(temp_output, temp_shape, input, in_shape, ndim, w->kernel->offsets, w->kernel->weights, w->k_elems, w->kernel);
        float ablated_loss = compute_loss(temp_output, output, out_total);
        w->kernel->weights[idx] = orig_weight; // Restore
        if (ablated_loss <= base_loss * 1.1f) { // Remove if ablation doesn't increase loss significantly
            w->selected_indices[i] = w->selected_indices[--w->selected_count];
            i--;
        }
        memcpy(temp_shape, out_shape, ndim * sizeof(int)); // Reset temp_shape
    }
    free(temp_output);
    free(temp_shape);
}

void temporal_wrapper_forward(TemporalWrapper *w, float *output, int *out_shape, const float *input, const int *in_shape, int ndim) {
    if (!w->formula_expr) {
        fprintf(stderr, "Formula not compiled, falling back to standard convolution\n");
        convn_forward_custom(output, out_shape, input, in_shape, ndim, w->kernel->offsets, w->kernel->weights, w->k_elems, w->kernel);
        return;
    }

    // Apply formula to adjust weights
    float *adjusted_weights = (float*)malloc(w->k_elems * sizeof(float));
    if (!adjusted_weights) {
        fprintf(stderr, "Failed to allocate adjusted weights\n");
        convn_forward_custom(output, out_shape, input, in_shape, ndim, w->kernel->offsets, w->kernel->weights, w->k_elems, w->kernel);
        return;
    }
    for (int i = 0; i < w->k_elems; ++i) {
        int is_selected = 0;
        for (int j = 0; j < w->selected_count; ++j) {
            if (w->selected_indices[j] == i) {
                is_selected = 1;
                break;
            }
        }
        if (is_selected) {
            float args[w->t_formula->var_count];
            for (int v = 0; v < w->t_formula->var_count; ++v) {
                if (strcmp(w->t_formula->vars[v], "w") == 0) args[v] = w->kernel->weights[i];
                else if (strcmp(w->t_formula->vars[v], "k") == 0) args[v] = w->k;
                else if (strcmp(w->t_formula->vars[v], "g") == 0) args[v] = w->grad_magnitudes[i];
                else args[v] = 0.0f; // Default for unknown variables
            }
            adjusted_weights[i] = exprset_eval(w->formula_expr, args);
        } else {
            adjusted_weights[i] = w->kernel->weights[i];
        }
    }

    // Perform convolution with adjusted weights
    convn_forward_custom(output, out_shape, input, in_shape, ndim, w->kernel->offsets, adjusted_weights, w->k_elems, w->kernel);
    free(adjusted_weights);
}

void temporal_wrapper_backward(TemporalWrapper *w, const float *input, const int *in_shape, int ndim, const float *d_output, const int *out_shape, float *d_input, float *d_weights) {
    if (!w->formula_expr) {
        fprintf(stderr, "Formula not compiled, falling back to standard backward pass\n");
        convn_backward_custom(input, in_shape, ndim, w->kernel->offsets, w->kernel->weights, w->k_elems, d_output, out_shape, d_input, d_weights, w->kernel);
        return;
    }

    // Compute gradients with respect to adjusted weights
    float *adjusted_weights = (float*)malloc(w->k_elems * sizeof(float));
    float *d_adjusted_weights = (float*)calloc(w->k_elems, sizeof(float));
    if (!adjusted_weights || !d_adjusted_weights) {
        fprintf(stderr, "Failed to allocate memory for backward pass\n");
        convn_backward_custom(input, in_shape, ndim, w->kernel->offsets, w->kernel->weights, w->k_elems, d_output, out_shape, d_input, d_weights, w->kernel);
        free(adjusted_weights); free(d_adjusted_weights);
        return;
    }
    w->k_grad = 0.0f;

    for (int i = 0; i < w->k_elems; ++i) {
        int is_selected = 0;
        for (int j = 0; j < w->selected_count; ++j) {
            if (w->selected_indices[j] == i) {
                is_selected = 1;
                break;
            }
        }
        if (is_selected) {
            float args[w->t_formula->var_count];
            for (int v = 0; v < w->t_formula->var_count; ++v) {
                if (strcmp(w->t_formula->vars[v], "w") == 0) args[v] = w->kernel->weights[i];
                else if (strcmp(w->t_formula->vars[v], "k") == 0) args[v] = w->k;
                else if (strcmp(w->t_formula->vars[v], "g") == 0) args[v] = w->grad_magnitudes[i];
                else args[v] = 0.0f;
            }
            adjusted_weights[i] = exprset_eval(w->formula_expr, args);
        } else {
            adjusted_weights[i] = w->kernel->weights[i];
        }
    }

    convn_backward_custom(input, in_shape, ndim, w->kernel->offsets, adjusted_weights, w->k_elems, d_output, out_shape, d_input, d_adjusted_weights, w->kernel);

    // Backpropagate gradients to original weights and k
    for (int i = 0; i < w->k_elems; ++i) {
        int is_selected = 0;
        for (int j = 0; j < w->selected_count; ++j) {
            if (w->selected_indices[j] == i) {
                is_selected = 1;
                break;
            }
        }
        if (is_selected) {
            // Gradient w.r.t. w: dL/dw = dL/dw_adj * dw_adj/dw
            float dw_adj_dw = 1.0f;
            for (int v = 0; v < w->t_formula->var_count; ++v) {
                if (strcmp(w->t_formula->vars[v], "w") == 0) {
                    dw_adj_dw = 1.0f + w->k * w->grad_magnitudes[i]; // Simplified derivative
                    break;
                }
            }
            d_weights[i] = d_adjusted_weights[i] * dw_adj_dw;
            // Gradient w.r.t. k: dL/dk = dL/dw_adj * dw_adj/dk
            float dw_adj_dk = w->kernel->weights[i] * w->grad_magnitudes[i]; // Simplified derivative
            for (int v = 0; v < w->t_formula->var_count; ++v) {
                if (strcmp(w->t_formula->vars[v], "k") == 0) {
                    w->k_grad += d_adjusted_weights[i] * dw_adj_dk;
                    break;
                }
            }
        } else {
            d_weights[i] = d_adjusted_weights[i];
        }
    }

    free(adjusted_weights);
    free(d_adjusted_weights);
}
