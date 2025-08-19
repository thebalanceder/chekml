#include "HS.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// Random double in [min, max]
static double rand_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// Clamp value within bounds
static double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Harmony Search optimizer following HarmonySearch.py logic
void HS_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    int num_vars = opt->dim;
    int memory_size = HS_MEMORY_SIZE;
    int max_iterations = HS_MAX_ITERATIONS;
    double harmony_memory_considering_rate = HS_HARMONY_MEMORY_CONSIDERING_RATE;
    double pitch_adjustment_rate = HS_PITCH_ADJUSTMENT_RATE;
    double bandwidth = HS_BANDWIDTH;

    // Allocate memory for harmony memory and fitness
    double** harmony_memory = (double**)malloc(memory_size * sizeof(double*));
    double* fitness = (double*)malloc(memory_size * sizeof(double));
    for (int i = 0; i < memory_size; i++) {
        harmony_memory[i] = (double*)malloc(num_vars * sizeof(double));
    }

    // Initialization: Random solutions for harmony memory
    for (int i = 0; i < memory_size; i++) {
        for (int d = 0; d < num_vars; d++) {
            double lower = opt->bounds[2 * d];
            double upper = opt->bounds[2 * d + 1];
            harmony_memory[i][d] = rand_uniform(lower, upper);
        }
    }

    // Evaluate initial fitness
    for (int i = 0; i < memory_size; i++) {
        fitness[i] = objective_function(harmony_memory[i]);
    }

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
        // Harmony improvisation
        for (int i = 0; i < num_vars; i++) {
            int idx = rand() % memory_size;
            double* harmony = harmony_memory[idx];
            double* new_harmony = (double*)malloc(num_vars * sizeof(double));

            // Harmony memory considering rate (HMCR)
            double rand_val = rand_uniform(0.0, 1.0);
            if (rand_val < harmony_memory_considering_rate) {
                // Copy from harmony memory
                memcpy(new_harmony, harmony, num_vars * sizeof(double));
            } else {
                // Generate random harmony
                for (int d = 0; d < num_vars; d++) {
                    double lower = opt->bounds[2 * d];
                    double upper = opt->bounds[2 * d + 1];
                    new_harmony[d] = rand_uniform(lower, upper);
                }
            }

            // Pitch adjustment rate (PAR)
            rand_val = rand_uniform(0.0, 1.0);
            if (rand_val < pitch_adjustment_rate) {
                // Adjust harmony (within bandwidth)
                for (int d = 0; d < num_vars; d++) {
                    double adjustment = rand_uniform(-bandwidth, bandwidth);
                    new_harmony[d] += adjustment;
                    // Clamp to bounds
                    double lower = opt->bounds[2 * d];
                    double upper = opt->bounds[2 * d + 1];
                    new_harmony[d] = clamp(new_harmony[d], lower, upper);
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

            free(new_harmony);
        }

        // Print progress
        printf("Iteration %d: Best Fitness = %f\n", iter + 1, best_fitness);
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

