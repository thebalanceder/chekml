#ifndef ALO_H
#define ALO_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // For malloc/free
#include <stdint.h>  // For uint32_t in random number generator
#include "generaloptimizer.h"  // Main optimizer header

// Optimization parameters
#define I_FACTOR_1 100.0       // Scaling factor for I after 10% iterations
#define I_FACTOR_2 1000.0      // Scaling factor for I after 50% iterations
#define I_FACTOR_3 10000.0     // Scaling factor for I after 75% iterations
#define I_FACTOR_4 100000.0    // Scaling factor for I after 90% iterations
#define I_FACTOR_5 1000000.0   // Scaling factor for I after 95% iterations
#define ROULETTE_EPSILON 1e-10 // Small value to avoid division by zero in roulette wheel

// Cache alignment
#define CACHE_LINE_SIZE 64

// Fast random number generator (Xorshift)
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline double alo_rand_double(uint32_t *state, double min, double max) {
    return min + (max - min) * (xorshift32(state) / (double)0xFFFFFFFF);
}

// ALO Algorithm Phases
void initialize_populations(Optimizer *opt, double *antlion_positions, uint32_t *rng_state);
void random_walk_phase(Optimizer *opt, int t, double *antlion, double *walk_buffer, uint32_t *rng_state);
void update_ant_positions(Optimizer *opt, int t, double *antlion_positions, double *walk_buffer, double *weights, uint32_t *rng_state);
void update_antlions_phase(Optimizer *opt, double *antlion_positions, double *combined_fitness, int *indices);
int roulette_wheel_selection(double *weights, int size, uint32_t *rng_state);
static void quicksort_with_indices(double *arr, int *indices, int low, int high);

// Optimization Execution
void ALO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // ALO_H
