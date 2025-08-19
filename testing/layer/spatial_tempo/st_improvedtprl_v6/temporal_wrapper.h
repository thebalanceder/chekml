#ifndef TEMPORAL_WRAPPER_H
#define TEMPORAL_WRAPPER_H

#include "kof_parser.h"
#include "formula_eval.h"
#include "temporal_parser.h"

typedef struct {
    KOFKernel *kernel;           // Pointer to the original kernel
    int k_elems;                 // Number of weights (same as kernel->k_elems)
    float *weight_history;       // History of weights [k_elems * history_length]
    float *gradient_history;     // History of gradients [k_elems * history_length]
    int history_count;           // Current number of stored history entries
    float *grad_magnitudes;      // Average gradient magnitudes [k_elems]
    float *memory_state;         // Long-term memory state [k_elems]
    int *selected_indices;       // Indices of selected weights [k_elems]
    int selected_count;          // Number of selected weights
    TemporalFormula *t_formula;  // Parsed temporal formula
    ExprSet *formula_expr;       // Compiled temporal formula
    ExprSet *memory_expr;        // Compiled long-term memory function
    ExprSet *testing_lhs_expr;   // Compiled LHS of testing formula
    ExprSet *testing_rhs_expr;   // Compiled RHS of testing formula
    float k[MAX_LEARNABLE_PARAMS]; // Learnable parameters
    float k_grad[MAX_LEARNABLE_PARAMS]; // Gradients of learnable parameters
    int k_count;                 // Number of learnable parameters
} TemporalWrapper;

TemporalWrapper *temporal_wrapper_create(KOFKernel *kernel, const char *tprl_filename);
void temporal_wrapper_free(TemporalWrapper *w);
void temporal_wrapper_update(TemporalWrapper *w, const float *gradients, float loss, const float *output, const int *out_shape, const float *input, const int *in_shape, int ndim);
void temporal_wrapper_forward(TemporalWrapper *w, float *output, int *out_shape, const float *input, const int *in_shape, int ndim);
void temporal_wrapper_backward(TemporalWrapper *w, const float *input, const int *in_shape, int ndim, const float *d_output, const int *out_shape, float *d_input, float *d_weights);

#endif // TEMPORAL_WRAPPER_H
