#ifndef FORMULA_EVAL_H
#define FORMULA_EVAL_H

#include "kof_parser.h"
#include "tinyexpr.h"

#define MAX_VARS 10

typedef struct {
    te_expr *expr;
    te_variable vars[MAX_VARS];
    double values[MAX_VARS];
    int nvars;
} ExprSet;

typedef struct {
    // Global fallback formulas
    ExprSet *op;
    ExprSet *input_encode;
    ExprSet *kernel_encode;

    // Per-tap formulas (can be NULL if not overridden)
    ExprSet **tap_ops;           // [k_elems]
    ExprSet **tap_input_encodes; // [k_elems]
    ExprSet **tap_kernel_encodes;// [k_elems]
    int k_elems;
} FormulaSet;

// Compile a single formula into an expression evaluator
ExprSet *exprset_compile(const char *expr_str, const char **vars, int varcount);
float exprset_eval(ExprSet *e, float *args);
void exprset_free(ExprSet *e);

// Compile all formulas for kernel
FormulaSet *formula_create(const KOFKernel *k);
void formula_free(FormulaSet *f);

#endif // FORMULA_EVAL_H
