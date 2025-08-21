#ifndef DEA_H
#define DEA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "generaloptimizer.h"

// DEA Optimization Parameters
#define DEA_POPULATION_SIZE 50
#define DEA_MAX_LOOPS 100
#define DEA_CONVERGENCE_POWER 1.0f
#define DEA_EFFECTIVE_RADIUS_FACTOR 0.25f
#define DEA_PROBABILITY_THRESHOLD 0.1f
#define DEA_ALTERNATIVES_PER_DIM 100

// Aligned data structure for cache efficiency
typedef struct {
    float *values;        // Discretized alternative values
    float *accum_fitness; // Accumulative fitness for alternatives
    float *probabilities; // Probabilities for alternatives
    int size;             // Number of alternatives
    int pad[3];           // Padding for 64-byte alignment
} DimensionData __attribute__((aligned(64)));

// Global DEA data
typedef struct {
    DimensionData *dim_data; // Per-dimension alternatives and fitness
    float effective_radius;  // Effective radius for fitness distribution
    uint32_t rng_state;      // Xorshift RNG state
    int dim;                 // Number of dimensions
    int pop_size;            // Population size
    int pad[3];              // Padding for 64-byte alignment
} DEAData __attribute__((aligned(64)));

// DEA Algorithm Phases
void initialize_locations(Optimizer *opt, DEAData *dea_data);
void calculate_accumulative_fitness(Optimizer *opt, float *fitness, DEAData *dea_data);
static inline float get_convergence_probability(int loop, int max_loops);
void update_probabilities(Optimizer *opt, int loop, float *fitness, DEAData *dea_data);
void select_new_locations(Optimizer *opt, DEAData *dea_data);
void DEA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // DEA_H
