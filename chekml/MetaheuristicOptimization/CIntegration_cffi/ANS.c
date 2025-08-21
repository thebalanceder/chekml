#include "ANS.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>  // AVX2 intrinsics

// SIMD-friendly random number generation
static inline double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// Clamp value within bounds
static inline double clamp(double x, double min, double max) {
    return (x < min) ? min : (x > max) ? max : x;
}

// AVX2 Version of fitness calculation
static inline double fitness_function(double* p, int num_vars) {
    __m256d sum = _mm256_setzero_pd();  // Initialize sum vector to zero

    // Process 4 elements at a time with AVX2
    int i = 0;
    for (; i + 3 < num_vars; i += 4) {
        // Load 4 values from p into AVX2 register
        __m256d values = _mm256_loadu_pd(&p[i]);

        // Square each value and accumulate the results in the sum
        sum = _mm256_add_pd(sum, _mm256_mul_pd(values, values));
    }

    // Horizontal sum of the 4 values in the AVX register
    __m128d low = _mm256_castpd256_pd128(sum);            // Low 2 elements
    __m128d high = _mm256_extractf128_pd(sum, 1);          // High 2 elements

    low = _mm_add_pd(low, high);                           // Sum the two parts
    low = _mm_hadd_pd(low, low);                           // Horizontal add to sum two doubles

    double result;
    _mm_store_sd(&result, low);                            // Store the result

    // Process remaining values if num_vars is not a multiple of 4
    for (; i < num_vars; i++) {
        result += p[i] * p[i];
    }

    return result;
}

// ANS optimizer following ANS.m logic with AVX2 optimizations
void ANS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    int num_vars = opt->dim;
    int num_neighborhoods = ANS_NUM_NEIGHBORHOODS;
    int max_iterations = opt->max_iter;
    double mutation_rate = ANS_MUTATION_RATE;

    // Allocate neighborhoods
    double** populations = (double**)malloc(num_neighborhoods * sizeof(double*));
    double* fitness = (double*)malloc(num_neighborhoods * sizeof(double));

    // Memory alignment for SIMD performance (32-byte alignment for AVX2)
    for (int i = 0; i < num_neighborhoods; i++) {
        populations[i] = (double*)aligned_alloc(32, num_vars * sizeof(double));  // 32-byte alignment
    }

    // Initialization: Random solutions for each neighborhood
    for (int i = 0; i < num_neighborhoods; i++) {
        for (int d = 0; d < num_vars; d++) {
            double lower = opt->bounds[2 * d];
            double upper = opt->bounds[2 * d + 1];
            populations[i][d] = rand_uniform(lower, upper);
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iterations; iter++) {
        double best_fit = INFINITY;
        int best_idx = 0;

        // === Parallel fitness evaluation using AVX2 ===
        #pragma omp parallel for shared(populations, fitness)
        for (int i = 0; i < num_neighborhoods; i++) {
            double* p = populations[i];
            fitness[i] = fitness_function(p, num_vars);
        }

        // === Find the best fitness in the neighborhood (serial) ===
        for (int i = 0; i < num_neighborhoods; i++) {
            if (fitness[i] < best_fit) {
                best_fit = fitness[i];
                best_idx = i;
            }
        }

        #if VERBOSE
        printf("ANS Iteration %d: Best Fitness = %f\n", iter + 1, best_fit);
        #endif

        // === Move each neighborhood in parallel ===
        #pragma omp parallel for
        for (int i = 0; i < num_neighborhoods; i++) {
            int neighbor_index = rand() % num_neighborhoods;
            while (neighbor_index == i) {
                neighbor_index = rand() % num_neighborhoods;
            }

            double* p = populations[i];
            double* neighbor = populations[neighbor_index];

            for (int d = 0; d < num_vars; d++) {
                double direction = neighbor[d] - p[d];
                p[d] += mutation_rate * direction;

                // Clamp values within bounds
                double lower = opt->bounds[2 * d];
                double upper = opt->bounds[2 * d + 1];
                p[d] = clamp(p[d], lower, upper);
            }
        }

        // Final best solution assignment
        memcpy(opt->best_solution.position, populations[best_idx], num_vars * sizeof(double));
        opt->best_solution.fitness = best_fit;
    }

    // Clean-up
    for (int i = 0; i < num_neighborhoods; i++) {
        free(populations[i]);
    }
    free(populations);
    free(fitness);
}