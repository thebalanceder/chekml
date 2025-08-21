#ifndef MFO_H
#define MFO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define MFO_B_CONSTANT 1.0  // Spiral shape constant
#define MFO_A_INITIAL -1.0  // Initial 'a' parameter
#define MFO_A_FINAL -2.0    // Final 'a' parameter
#define TWO_PI 6.283185307179586  // 2 * M_PI
#define LOG_E 0.4342944819032518  // 1 / ln(10)

// MFO-specific data structure
typedef struct {
    ObjectiveFunction objective_function;
    int iteration;
    Solution *population;          // Current population
    double *fitness;              // Current fitness
    Solution *best_flames;        // Best flames
    double *best_flame_fitness;   // Fitness of best flames
    int *indices;                 // Sorting indices
    double *spiral_cache;         // Precomputed spiral values
    unsigned int rng_state;       // RNG state for fast random numbers
} MFOData;

// MFO Algorithm Phases
void mfo_update_flames(MFOData *restrict mfo_data, int population_size, int dim, int max_iter);
void mfo_update_moth_positions(MFOData *restrict mfo_data, int population_size, int dim, int max_iter);
void mfo_sort_population(MFOData *restrict mfo_data, int population_size, int dim);

// Optimization Execution
void MFO_optimize(void *opt, ObjectiveFunction objective_function);

// MFO Data Management
MFOData* mfo_create_data(int population_size, int dim);
void mfo_free_data(MFOData *mfo_data, int population_size);

#ifdef __cplusplus
}
#endif

#endif // MFO_H
