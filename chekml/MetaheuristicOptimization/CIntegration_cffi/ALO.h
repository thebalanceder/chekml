#ifndef ALO_H
#define ALO_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define I_FACTOR_1 100.0       // Scaling factor for I after 10% iterations
#define I_FACTOR_2 1000.0      // Scaling factor for I after 50% iterations
#define I_FACTOR_3 10000.0     // Scaling factor for I after 75% iterations
#define I_FACTOR_4 100000.0    // Scaling factor for I after 90% iterations
#define I_FACTOR_5 1000000.0   // Scaling factor for I after 95% iterations
#define ROULETTE_EPSILON 1e-10 // Small value to avoid division by zero in roulette wheel

// Random number generation (inline for performance)
static inline double rand_double_alo(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// ALO Algorithm Phases
void initialize_populations(Optimizer *opt);
void random_walk_phase(Optimizer *opt, int t, double *antlion, double *walk_buffer);
void update_ant_positions(Optimizer *opt, int t, double *antlion_positions, double *walk_buffer);
void update_antlions_phase(Optimizer *opt, double *antlion_positions);
int roulette_wheel_selection(double *weights, int size);
void quicksort_with_indices(double *arr, int *indices, int low, int high);

// Optimization Execution
void ALO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // ALO_H
