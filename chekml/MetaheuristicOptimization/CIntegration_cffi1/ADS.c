/* ADS.c - Optimized for CPU-Only Adaptive Dimension Search */
#include "ADS.h"
#include <string.h>
#include <float.h>

static double g_max = -INFINITY;
static double g_min = INFINITY;
static double best_value = INFINITY;
static double *best_solution = NULL;
static int collocation_points = 0;

static inline double evaluate(const double *x, int dim, double (*func)(double *)) {
    double val = func((double *)x);
    if (val > g_max) g_max = val;
    if (val < g_min) g_min = val;
    if (val < best_value) {
        best_value = val;
        memcpy(best_solution, x, sizeof(double) * dim);
    }
    return val;
}

static inline void generate_cc_points(double *points, int *num_points) {
    const int m = 9;
    *num_points = m;
    for (int i = 0; i < m; ++i) {
        double theta = M_PI * i / (m - 1);
        points[i] = 0.5 * (1.0 - cos(theta));
    }
}

static inline void interpolate_dimension(int dim_idx, int dim, const double *ref, const double *bounds, double (*func)(double *)) {
    double pts[9];
    int n;
    generate_cc_points(pts, &n);

    double scale = bounds[2 * dim_idx + 1] - bounds[2 * dim_idx];
    double offset = bounds[2 * dim_idx];

    for (int i = 0; i < n && collocation_points < MAX_COLLOCATION_POINTS; ++i) {
        double x[dim];
        memcpy(x, ref, sizeof(double) * dim);
        x[dim_idx] = offset + pts[i] * scale;
        evaluate(x, dim, func);
        ++collocation_points;
    }
}

void ADS_optimize(Optimizer *opt, double (*performance_function)(double *)) {
    int dim = opt->dim;
    const double *bounds = opt->bounds;

    double *ref = (double *)malloc(sizeof(double) * dim);
    best_solution = (double *)malloc(sizeof(double) * dim);

    for (int i = 0; i < dim; ++i)
        ref[i] = 0.5 * (bounds[2 * i] + bounds[2 * i + 1]);

    evaluate(ref, dim, performance_function);
    ++collocation_points;

    for (int i = 0; i < dim && collocation_points < MAX_COLLOCATION_POINTS; ++i) {
        interpolate_dimension(i, dim, ref, bounds, performance_function);
    }

    memcpy(opt->best_solution.position, best_solution, sizeof(double) * dim);
    opt->best_solution.fitness = best_value;

    free(ref);
    free(best_solution);
}

