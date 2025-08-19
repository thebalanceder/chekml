#ifndef WGMO_H
#define WGMO_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // For malloc/free
#include <stdio.h>   // For debugging/logging
#include <stdint.h>  // For uint64_t
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define WGMO_ALPHA 0.9
#define WGMO_BETA 0.1
#define WGMO_GAMMA 0.1

// WGMO-specific Xorshift random number generator state
typedef struct {
    uint64_t state;
} WGMOXorshiftState;

// WGMO Algorithm Phases
// Note: Caller is responsible for allocating and freeing population[i].position and best_solution.position
// Functions preserve the memory layout of population[i].position as allocated by general_init
void initialize_geese(Optimizer *opt, WGMOXorshiftState *rng);
void evaluate_and_sort_geese(Optimizer *opt, double (*objective_function)(double *), int *indices);
void update_geese_positions(Optimizer *opt, int *indices, double *rand_buffer, int rand_offset);

// Optimization Execution
void WGMO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // WGMO_H
