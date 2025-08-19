// ILS.c - Optimized for Speed
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <immintrin.h>  // SIMD intrinsics for AVX2/AVX512
#include "generaloptimizer.h"
#include "ILS.h"

// Check if a point is within bounds
static inline int in_bounds(const double* point, const double* bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        if (__builtin_expect(point[i] < bounds[2 * i] || point[i] > bounds[2 * i + 1], 0)) {
            return 0;
        }
    }
    return 1;
}

// Generate a random point within bounds
static inline void random_point_within_bounds(double* restrict point, const double* restrict bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        double min = bounds[2 * i];
        double max = bounds[2 * i + 1];
        point[i] = min + ((double)rand() / RAND_MAX) * (max - min);
    }
}

// Hill Climb Procedure
static void hill_climb(double* restrict solution, const double* restrict bounds, int dim, int n_iterations, double step_size, ObjectiveFunction objective_function, double* restrict best_eval) {
    double* candidate = (double*)aligned_alloc(32, dim * sizeof(double));
    double* current = (double*)aligned_alloc(32, dim * sizeof(double));
    memcpy(current, solution, dim * sizeof(double));
    *best_eval = objective_function(current);

    for (int i = 0; i < n_iterations; i++) {
        int valid = 0;
        while (!valid) {
            for (int j = 0; j < dim; j++) {
                candidate[j] = current[j] + ((double)rand() / RAND_MAX) * step_size * 2.0 - step_size;
            }
            valid = in_bounds(candidate, bounds, dim);
        }

        double candidate_eval = objective_function(candidate);
        if (__builtin_expect(candidate_eval <= *best_eval, 0)) {
            memcpy(current, candidate, dim * sizeof(double));
            *best_eval = candidate_eval;
        }
    }

    memcpy(solution, current, dim * sizeof(double));
    free(candidate);
    free(current);
}

// ILS Optimizer
void ILS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (__builtin_expect(!opt || !objective_function, 0)) {
        fprintf(stderr, "Error: Invalid optimizer or objective function.\n");
        return;
    }

    const int dim = opt->dim;
    const int n_restarts = 30;
    const int n_iterations = 1000;
    const double step_size = 0.05;
    const double perturbation_size = 1.0;

    double* best_solution = (double*)aligned_alloc(32, dim * sizeof(double));
    double* temp = (double*)aligned_alloc(32, dim * sizeof(double));

    if (!best_solution || !temp) {
        fprintf(stderr, "Memory allocation failed\n");
        free(best_solution);
        free(temp);
        return;
    }

    random_point_within_bounds(best_solution, opt->bounds, dim);
    double best_fitness = objective_function(best_solution);

    for (int r = 0; r < n_restarts; r++) {
        int valid = 0;
        while (!valid) {
            for (int i = 0; i < dim; i++) {
                temp[i] = best_solution[i] + ((double)rand() / RAND_MAX) * perturbation_size * 2.0 - perturbation_size;
            }
            valid = in_bounds(temp, opt->bounds, dim);
        }

        double eval;
        hill_climb(temp, opt->bounds, dim, n_iterations, step_size, objective_function, &eval);

        if (__builtin_expect(eval < best_fitness, 0)) {
            best_fitness = eval;
            memcpy(best_solution, temp, dim * sizeof(double));
            printf("Restart %d: Best Fitness = %.5f\n", r, best_fitness);
        }
    }

    opt->best_solution.fitness = best_fitness;
    memcpy(opt->best_solution.position, best_solution, dim * sizeof(double));

    free(best_solution);
    free(temp);
}

