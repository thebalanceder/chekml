// ILS.c - Deep Optimized for Ultimate CPU-Specific Speed
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <immintrin.h>  // SIMD intrinsics for AVX2/AVX512
#include <stdint.h>
#include <time.h>
#include "generaloptimizer.h"
#include "ILS.h"

#define ALIGNMENT 64

// Fast random double in range [-1.0, 1.0]
static inline double fast_rand_signed() {
    return ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
}

// Bounds check with loop unrolling
static inline int in_bounds_unrolled(const double* __restrict point, const double* __restrict bounds, int dim) {
    for (int i = 0; i < dim; i += 2) {
        if (point[i] < bounds[2 * i] || point[i] > bounds[2 * i + 1]) return 0;
        if (i + 1 < dim && (point[i + 1] < bounds[2 * (i + 1)] || point[i + 1] > bounds[2 * (i + 1) + 1])) return 0;
    }
    return 1;
}

// Generate random vector within bounds using SIMD where possible
static inline void random_point_within_bounds_simd(double* __restrict point, const double* __restrict bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        double min = bounds[2 * i];
        double max = bounds[2 * i + 1];
        point[i] = min + ((double)rand() / RAND_MAX) * (max - min);
    }
}

// Hill Climb Optimized
static void hill_climb_fast(double* __restrict solution, const double* __restrict bounds, int dim, int n_iterations, double step_size, ObjectiveFunction obj_func, double* __restrict best_eval) {
    double* __restrict candidate = aligned_alloc(ALIGNMENT, dim * sizeof(double));
    double* __restrict current = aligned_alloc(ALIGNMENT, dim * sizeof(double));
    memcpy(current, solution, dim * sizeof(double));
    *best_eval = obj_func(current);

    for (int i = 0; i < n_iterations; i++) {
        int valid = 0;
        while (!valid) {
            for (int j = 0; j < dim; ++j) {
                candidate[j] = current[j] + fast_rand_signed() * step_size;
            }
            valid = in_bounds_unrolled(candidate, bounds, dim);
        }

        double eval = obj_func(candidate);
        if (eval <= *best_eval) {
            *best_eval = eval;
            memcpy(current, candidate, dim * sizeof(double));
        }
    }

    memcpy(solution, current, dim * sizeof(double));
    free(candidate);
    free(current);
}

// Main ILS optimization routine
void ILS_optimize(Optimizer* opt, ObjectiveFunction obj_func) {
    if (__builtin_expect(!opt || !obj_func, 0)) {
        fprintf(stderr, "[ILS Error] Invalid optimizer or objective function.\n");
        return;
    }

    const int dim = opt->dim;
    const int restarts = 30;
    const int iterations = 1000;
    const double step_size = 0.05;
    const double perturb = 1.0;

    double* best_sol = aligned_alloc(ALIGNMENT, dim * sizeof(double));
    double* temp_sol = aligned_alloc(ALIGNMENT, dim * sizeof(double));

    if (!best_sol || !temp_sol) {
        fprintf(stderr, "[ILS Error] Memory allocation failed.\n");
        free(best_sol);
        free(temp_sol);
        return;
    }

    random_point_within_bounds_simd(best_sol, opt->bounds, dim);
    double best_fit = obj_func(best_sol);

    for (int r = 0; r < restarts; r++) {
        int valid = 0;
        while (!valid) {
            for (int i = 0; i < dim; ++i) {
                temp_sol[i] = best_sol[i] + fast_rand_signed() * perturb;
            }
            valid = in_bounds_unrolled(temp_sol, opt->bounds, dim);
        }

        double eval;
        hill_climb_fast(temp_sol, opt->bounds, dim, iterations, step_size, obj_func, &eval);

        if (eval < best_fit) {
            best_fit = eval;
            memcpy(best_sol, temp_sol, dim * sizeof(double));
            printf("Restart %d: Best Fitness = %.6f\n", r + 1, best_fit);
        }
    }

    opt->best_solution.fitness = best_fit;
    memcpy(opt->best_solution.position, best_sol, dim * sizeof(double));

    free(best_sol);
    free(temp_sol);
}

