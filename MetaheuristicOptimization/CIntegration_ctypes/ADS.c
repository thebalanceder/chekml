/* ADS.c - Implementation of Adaptive Dimension Search (ADS) */
#include "ADS.h"
#include <string.h>
#include <float.h>

static double g_max = -INFINITY;
static double g_min = INFINITY;
static double best_value = INFINITY;
static double *best_solution = NULL;
static int collocation_points = 0;

// Utility: Random in [0,1]
static inline double rand_unit() {
    return (double)rand() / RAND_MAX;
}

// Evaluate function and track best
static double evaluate(double *x, int dim, double (*func)(double *)) {
    double val = func(x);
    if (val > g_max) g_max = val;
    if (val < g_min) g_min = val;
    if (val < best_value) {
        best_value = val;
        memcpy(best_solution, x, sizeof(double) * dim);
    }
    return val;
}

// Clenshaw-Curtis points generator for 1D
static void clenshaw_curtis_points(int level, double *points, int *num_points) {
    int m = (1 << level) + 1;
    *num_points = m;
    for (int i = 0; i < m; i++) {
        double theta = M_PI * i / (m - 1);
        points[i] = 0.5 * (1.0 - cos(theta));
    }
}

// Adaptive interpolation for 1D (mock-up)
static void interpolate_dimension(int dim_idx, int dim, double *ref, double *bounds, double (*func)(double *)) {
    double pts[9]; int n;
    clenshaw_curtis_points(3, pts, &n);

    for (int i = 0; i < n && collocation_points < MAX_COLLOCATION_POINTS; i++) {
        double x[dim];
        memcpy(x, ref, sizeof(double) * dim);
        x[dim_idx] = bounds[2 * dim_idx] + pts[i] * (bounds[2 * dim_idx + 1] - bounds[2 * dim_idx]);
        evaluate(x, dim, func);
        collocation_points++;
    }
}

// Main optimization function
void ADS_optimize(Optimizer *opt, double (*performance_function)(double *)) {
    int dim = opt->dim;
    double *bounds = opt->bounds;
    double *ref = malloc(sizeof(double) * dim);
    best_solution = malloc(sizeof(double) * dim);

    for (int i = 0; i < dim; i++) {
        ref[i] = (bounds[2 * i] + bounds[2 * i + 1]) / 2.0;
    }

    // Zeroth-order component (reference point only)
    evaluate(ref, dim, performance_function);
    collocation_points++;

    // First-order components
    for (int i = 0; i < dim && collocation_points < MAX_COLLOCATION_POINTS; i++) {
        interpolate_dimension(i, dim, ref, bounds, performance_function);
    }

    for (int j = 0; j < dim; j++) {
        opt->best_solution.position[j] = best_solution[j];
    }
    opt->best_solution.fitness = best_value;

    free(ref);
    free(best_solution);
}
