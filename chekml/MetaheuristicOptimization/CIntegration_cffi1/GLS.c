#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <immintrin.h>  // AVX/AVX2 intrinsics
#include "generaloptimizer.h"
#include "GLS.h"

// SIMD-optimized function to compute the augmented objective (g(s) + λ * sum(l_i(s) * c_i))
double evaluate_augmented_objective(Optimizer* opt, ObjectiveFunction objective_function, double* solution, double* penalties, double lambda, int dim) {
    double g = objective_function(solution);
    double penalty_term = 0.0;

    // Use AVX2 intrinsics for vectorized computation of penalty term
    __m256d sum = _mm256_setzero_pd();  // Initialize a 256-bit vector to store the sum

    #pragma omp parallel for
    for (int i = 0; i < dim; i += 4) {  // Process 4 elements per iteration
        __m256d sol = _mm256_loadu_pd(&solution[i]);  // Load 4 solution values into the vector
        __m256d pen = _mm256_loadu_pd(&penalties[i]);  // Load 4 penalty values into the vector
        __m256d product = _mm256_mul_pd(sol, pen);  // Multiply solution and penalty elements
        sum = _mm256_add_pd(sum, product);  // Add the product to the sum
    }

    // Horizontal sum the results from the SIMD vector
    double result[4];
    _mm256_storeu_pd(result, sum);
    penalty_term = result[0] + result[1] + result[2] + result[3];

    return g + lambda * penalty_term;
}

// Optimized GLS function with CPU-specific optimizations
void GLS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (!opt || !objective_function) {
        fprintf(stderr, "Error: Invalid optimizer or objective function.\n");
        return;
    }

    int dim = opt->dim;
    int max_iter = opt->max_iter;

    double lambda = 0.1;
    double* penalties = (double*)calloc(dim, sizeof(double));
    if (!penalties) {
        fprintf(stderr, "Memory allocation failed for penalties\n");
        return;
    }

    double* current_position = (double*)malloc(dim * sizeof(double));
    double current_fitness = DBL_MAX;
    double best_fitness = DBL_MAX;

    if (!current_position) {
        fprintf(stderr, "Memory allocation failed for current_position\n");
        free(penalties);
        return;
    }

    double* best_solution = (double*)malloc(dim * sizeof(double));
    if (!best_solution) {
        fprintf(stderr, "Memory allocation failed for best_solution\n");
        free(penalties);
        free(current_position);
        return;
    }

    // Random initial solution
    for (int i = 0; i < dim; i++) {
        current_position[i] = opt->bounds[2 * i] + ((double)rand() / RAND_MAX) * (opt->bounds[2 * i + 1] - opt->bounds[2 * i]);
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iter; iter++) {
        current_fitness = evaluate_augmented_objective(opt, objective_function, current_position, penalties, lambda, dim);

        if (current_fitness < best_fitness) {
            best_fitness = current_fitness;
            memcpy(best_solution, current_position, dim * sizeof(double));
        }

        // Controlled local search with SIMD unrolling
        #pragma omp parallel for
        for (int i = 0; i < dim; i++) {
            double perturbation = (rand() / (double)RAND_MAX) * 0.2 - 0.1;
            current_position[i] += perturbation;
            current_position[i] = fmin(fmax(current_position[i], opt->bounds[2 * i]), opt->bounds[2 * i + 1]);
        }

        // Feature indicators and penalty updates using AVX2 for parallel processing
        double* feature_indicators = (double*)malloc(dim * sizeof(double));
        if (!feature_indicators) {
            fprintf(stderr, "Memory allocation failed for feature_indicators\n");
            free(penalties);
            free(current_position);
            free(best_solution);
            return;
        }

        #pragma omp parallel for
        for (int i = 0; i < dim; i++) {
            feature_indicators[i] = (current_position[i] > 0.5) ? 1.0 : 0.0;
        }

        // Update penalties using SIMD intrinsics for vectorized operations
        #pragma omp parallel for
        for (int i = 0; i < dim; i++) {
            double utility = feature_indicators[i] * opt->bounds[2 * i] / (1 + penalties[i]);
            if (utility > 0.5) {
                // Prevent runaway penalties by controlling the update rate
                penalties[i] += fmin(1.0, 0.1);
            }
        }

        free(feature_indicators);

        // Save the history of the best solution
        opt->best_solution.fitness = best_fitness;
        memcpy(opt->best_solution.position, best_solution, dim * sizeof(double));

        // Print progress every 10 iterations
        if ((iter + 1) % 10 == 0) {
            printf("Iteration: %d | Best Fitness: %lf\n", iter + 1, best_fitness);
        }
    }

    // Clean up
    free(penalties);
    free(current_position);
    free(best_solution);
}

