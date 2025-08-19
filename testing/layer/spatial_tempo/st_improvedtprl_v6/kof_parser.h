#pragma once
#include <stdio.h>

typedef struct {
    int ndim;
    int k_elems;
    int anchor_is_explicit;   // 0: center/zero, 1: tuple
    int *anchor;              // ndim ints if explicit, else NULL
    enum { KOF_NORM_NONE, KOF_NORM_L1, KOF_NORM_L2, KOF_NORM_SUM1 } norm;
    float scale;
    int *offsets;             // length = k_elems * ndim
    float *weights;           // length = k_elems

    // Global formulas (optional fallback)
    char *operation_formula;      // e.g., "x * w"
    char *input_encode_formula;   // e.g., "log(1 + x)"
    char *kernel_encode_formula;  // e.g., "w * (1 + dx*dx + dy*dy)"

    // ✅ Per-tap formula overrides (new fields)
    char **tap_op_formulas;       // length = k_elems, NULL or string
    char **tap_input_formulas;    // length = k_elems
    char **tap_kernel_formulas;   // length = k_elems
} KOFKernel;

int kof_load(const char *path, KOFKernel *out); // returns 0 on success
void kof_free(KOFKernel *k);

