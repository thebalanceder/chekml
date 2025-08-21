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

// 🔧 Optimization parameters
#define MFO_B_CONSTANT 1.0  // Spiral shape constant (b in Eq. 3.12)
#define MFO_A_INITIAL -1.0  // Initial value of 'a' parameter
#define MFO_A_FINAL -2.0    // Final value of 'a' parameter

// 🌌 MFO-specific data structure
typedef struct {
    ObjectiveFunction objective_function;  // Function to optimize
    int iteration;                        // Current iteration
    Solution *previous_population;        // Previous population for flame update
    double *previous_fitness;             // Fitness of previous population
    Solution *best_flames;                // Best flames
    double *best_flame_fitness;           // Fitness of best flames
    Solution *sorted_population;          // Pre-allocated buffer for sorted population
    double *sorted_fitness;               // Pre-allocated buffer for sorted fitness
    int *indices;                         // Pre-allocated buffer for sorting indices
    Solution *temp_population;            // Pre-allocated buffer for combined population
    double *temp_fitness;                 // Pre-allocated buffer for combined fitness
    int *temp_indices;                    // Pre-allocated buffer for combined indices
} MFOData;

// 🌌 MFO Algorithm Phases
void mfo_update_flames(Optimizer *opt, MFOData *mfo_data);
void mfo_update_moth_positions(Optimizer *opt, MFOData *mfo_data);
void mfo_sort_population(Optimizer *opt, MFOData *mfo_data);

// 🚀 Optimization Execution
void MFO_optimize(void *opt, ObjectiveFunction objective_function);

// 🚧 MFO Data Management
MFOData* mfo_create_data(int population_size, int dim);
void mfo_free_data(MFOData *mfo_data, int population_size);

#ifdef __cplusplus
}
#endif

#endif // MFO_H
