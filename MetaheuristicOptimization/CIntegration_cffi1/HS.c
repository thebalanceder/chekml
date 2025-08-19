#include "HS.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>  // For SIMD (e.g., AVX2/AVX512)
#include <limits.h>     // For UINT_MAX
#include <omp.h>

// Constants for memory alignment and random number generation
#define ALIGNMENT 32    // Align memory to 32-byte boundary for SIMD
#define RNG_SEED 123456789  // Pseudo-random number generator seed for reproducibility

// Optimized PRNG: Using a simple but fast Xorshift32 algorithm for generating random numbers
static inline unsigned int xorshift32(unsigned int* state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

// Random double in [min, max] using optimized RNG (SIMD for batches)
static inline void rand_uniform_batch(unsigned int* rng_state, double* out, double min, double max, int batch_size) {
    #pragma omp simd
    for (int i = 0; i < batch_size; i++) {
        out[i] = min + ((double)(xorshift32(rng_state)) / UINT_MAX) * (max - min);
    }
}

// rand_uniform wrapper to use with existing code
static inline double rand_uniform(unsigned int* rng_state, double min, double max) {
    double out;
    rand_uniform_batch(rng_state, &out, min, max, 1);
    return out; // Return the first (and only) element of the generated batch
}

// Clamp value within bounds (branchless version)
static inline double clamp(double x, double min, double max) {
    double range = max - min;
    double tmp = (x < min) ? min : (x > max) ? max : x;
    return tmp;
}

// SIMD-optimized function to evaluate the objective for a batch of harmonies
static inline void evaluate_objective_batch(ObjectiveFunction objective_function, double** harmonies, double* fitness, int num_vars, int memory_size) {
    #pragma omp parallel for
    for (int i = 0; i < memory_size; i++) {
        fitness[i] = objective_function(harmonies[i]);
    }
}

// Harmony Search optimizer following HarmonySearch.py logic
void HS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    int num_vars = opt->dim;
    int memory_size = HS_MEMORY_SIZE;
    int max_iterations = HS_MAX_ITERATIONS;
    double harmony_memory_considering_rate = HS_HARMONY_MEMORY_CONSIDERING_RATE;
    double pitch_adjustment_rate = HS_PITCH_ADJUSTMENT_RATE;
    double bandwidth = HS_BANDWIDTH;

    // Allocate memory for harmony memory and fitness (aligned to cache line boundary)
    double** harmony_memory = (double**)aligned_alloc(ALIGNMENT, memory_size * sizeof(double*));
    double* fitness = (double*)aligned_alloc(ALIGNMENT, memory_size * sizeof(double));

    for (int i = 0; i < memory_size; i++) {
        harmony_memory[i] = (double*)aligned_alloc(ALIGNMENT, num_vars * sizeof(double));
    }

    unsigned int rng_state = RNG_SEED;  // Initialize PRNG state

    // Initialize harmonies randomly (using SIMD for better random generation if possible)
    #pragma omp parallel for
    for (int i = 0; i < memory_size; i++) {
        double* harmony = harmony_memory[i];
        rand_uniform_batch(&rng_state, harmony, opt->bounds[0], opt->bounds[1], num_vars);
    }

    // Evaluate initial fitness
    evaluate_objective_batch(objective_function, harmony_memory, fitness, num_vars, memory_size);

    // Identify best solution
    int best_idx = 0;
    for (int i = 1; i < memory_size; i++) {
        if (fitness[i] < fitness[best_idx]) {
            best_idx = i;
        }
    }

    double* best_solution = (double*)malloc(num_vars * sizeof(double));
    memcpy(best_solution, harmony_memory[best_idx], num_vars * sizeof(double));
    double best_fitness = fitness[best_idx];

    // Main optimization loop
    for (int iter = 0; iter < max_iterations; iter++) {
        // Harmony improvisation with SIMD optimizations
        #pragma omp parallel for
        for (int i = 0; i < memory_size; i++) {
            double* harmony = harmony_memory[i];
            double* new_harmony = harmony_memory[i]; // Reuse memory to avoid unnecessary allocations

            // Harmony memory considering rate (HMCR)
            double rand_val = rand_uniform(&rng_state, 0.0, 1.0);
            if (rand_val < harmony_memory_considering_rate) {
                // Copy from harmony memory (no SIMD needed, simple copy)
                memcpy(new_harmony, harmony, num_vars * sizeof(double));
            } else {
                // Generate random harmony (can be optimized with SIMD for generating batches)
                rand_uniform_batch(&rng_state, new_harmony, opt->bounds[0], opt->bounds[1], num_vars);
            }

            // Pitch adjustment rate (PAR)
            rand_val = rand_uniform(&rng_state, 0.0, 1.0);
            if (rand_val < pitch_adjustment_rate) {
                // SIMD-optimized pitch adjustment (applies to all dimensions)
                #pragma omp simd
                for (int d = 0; d < num_vars; d++) {
                    double adjustment = rand_uniform(&rng_state, -bandwidth, bandwidth);
                    new_harmony[d] += adjustment;
                    // Clamp to bounds
                    new_harmony[d] = clamp(new_harmony[d], opt->bounds[2 * d], opt->bounds[2 * d + 1]);
                }
            }

            // Evaluate new harmony
            double new_fitness = objective_function(new_harmony);

            // Replace the worst harmony if better
            if (new_fitness < fitness[best_idx]) {
                memcpy(harmony_memory[best_idx], new_harmony, num_vars * sizeof(double));
                fitness[best_idx] = new_fitness;
                best_fitness = new_fitness;
                memcpy(best_solution, new_harmony, num_vars * sizeof(double));
            }
        }

        // Print progress (only at every 100 iterations to reduce IO overhead)
        if (iter % 100 == 0) {
            printf("Iteration %d: Best Fitness = %f\n", iter + 1, best_fitness);
        }
    }

    // Save the best solution
    memcpy(opt->best_solution.position, best_solution, num_vars * sizeof(double));
    opt->best_solution.fitness = best_fitness;

    // Clean up
    free(best_solution);
    for (int i = 0; i < memory_size; i++) {
        free(harmony_memory[i]);
    }
    free(harmony_memory);
    free(fitness);
}
