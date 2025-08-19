#include "formula_eval.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "tinyexpr.h"

static const char *default_op = "x * w";
static const char *default_input_encode = "x";
static const char *default_kernel_encode = "w";

ExprSet *exprset_compile(const char *expr_str, const char **vars, int varcount) {
    ExprSet *e = (ExprSet*)calloc(1, sizeof(ExprSet));
    e->nvars = varcount;
    for (int i = 0; i < varcount; ++i) {
        e->vars[i].name = vars[i];
        e->vars[i].address = &e->values[i];
        e->vars[i].type = TE_VARIABLE;
    }
    int err = 0;
    e->expr = te_compile(expr_str, e->vars, varcount, &err);
    if (!e->expr) {
        fprintf(stderr, "Failed to compile formula: %s (error=%d)\n", expr_str, err);
        free(e);
        return NULL;
    }
    return e;
}

float exprset_eval(ExprSet *e, float *args) {
    for (int i = 0; i < e->nvars; ++i) {
        e->values[i] = args[i];
    }
    return (float)te_eval(e->expr);
}

void exprset_free(ExprSet *e) {
    if (!e) return;
    te_free(e->expr);
    free(e);
}

FormulaSet *formula_create(const KOFKernel *k) {
    FormulaSet *f = (FormulaSet*)calloc(1, sizeof(FormulaSet));
    f->k_elems = k->k_elems;

    const char *op_expr = k->operation_formula ? k->operation_formula : default_op;
    const char *enc_in = k->input_encode_formula ? k->input_encode_formula : default_input_encode;
    const char *enc_k  = k->kernel_encode_formula ? k->kernel_encode_formula : default_kernel_encode;

    const char *op_vars[] = {"x", "w"};
    const char *in_vars[] = {"x"};

    const char *dims[8] = {"dx", "dy", "dz", "du", "dv", "dw", "da", "db"};
    const char **k_vars = (const char**)malloc((1 + k->ndim) * sizeof(char*));
    k_vars[0] = "w";
    for (int i = 0; i < k->ndim; ++i) k_vars[i+1] = dims[i];

    // Compile global formulas
    f->op = exprset_compile(op_expr, op_vars, 2);
    if (!f->op) {
        fprintf(stderr, "[formula_create] Failed to compile global operation formula: %s\n", op_expr);
        free(k_vars); formula_free(f); return NULL;
    }

    f->input_encode = exprset_compile(enc_in, in_vars, 1);
    if (!f->input_encode) {
        fprintf(stderr, "[formula_create] Failed to compile global input_encode formula: %s\n", enc_in);
        free(k_vars); formula_free(f); return NULL;
    }

    f->kernel_encode = exprset_compile(enc_k, k_vars, 1 + k->ndim);
    if (!f->kernel_encode) {
        fprintf(stderr, "[formula_create] Failed to compile global kernel_encode formula: %s\n", enc_k);
        free(k_vars); formula_free(f); return NULL;
    }

    // Compile per-tap formulas
    if (k->tap_op_formulas || k->tap_input_formulas || k->tap_kernel_formulas) {
        f->tap_ops = (ExprSet**)calloc(k->k_elems, sizeof(ExprSet*));
        f->tap_input_encodes = (ExprSet**)calloc(k->k_elems, sizeof(ExprSet*));
        f->tap_kernel_encodes = (ExprSet**)calloc(k->k_elems, sizeof(ExprSet*));

        for (int i = 0; i < k->k_elems; ++i) {
            if (k->tap_op_formulas && k->tap_op_formulas[i]) {
                f->tap_ops[i] = exprset_compile(k->tap_op_formulas[i], op_vars, 2);
                if (!f->tap_ops[i]) {
                    fprintf(stderr, "[formula_create] Failed to compile tap %d op formula: %s\n", i, k->tap_op_formulas[i]);
                }
            }
            if (k->tap_input_formulas && k->tap_input_formulas[i]) {
                f->tap_input_encodes[i] = exprset_compile(k->tap_input_formulas[i], in_vars, 1);
                if (!f->tap_input_encodes[i]) {
                    fprintf(stderr, "[formula_create] Failed to compile tap %d input_encode formula: %s\n", i, k->tap_input_formulas[i]);
                }
            }
            if (k->tap_kernel_formulas && k->tap_kernel_formulas[i]) {
                f->tap_kernel_encodes[i] = exprset_compile(k->tap_kernel_formulas[i], k_vars, 1 + k->ndim);
                if (!f->tap_kernel_encodes[i]) {
                    fprintf(stderr, "[formula_create] Failed to compile tap %d kernel_encode formula: %s\n", i, k->tap_kernel_formulas[i]);
                }
            }
        }
    }

    free(k_vars);
    return f;
}

void formula_free(FormulaSet *f) {
    if (!f) return;
    exprset_free(f->op);
    exprset_free(f->input_encode);
    exprset_free(f->kernel_encode);

    if (f->tap_ops) {
        for (int i = 0; i < f->k_elems; ++i) exprset_free(f->tap_ops[i]);
        free(f->tap_ops);
    }
    if (f->tap_input_encodes) {
        for (int i = 0; i < f->k_elems; ++i) exprset_free(f->tap_input_encodes[i]);
        free(f->tap_input_encodes);
    }
    if (f->tap_kernel_encodes) {
        for (int i = 0; i < f->k_elems; ++i) exprset_free(f->tap_kernel_encodes[i]);
        free(f->tap_kernel_encodes);
    }

    free(f);
}
