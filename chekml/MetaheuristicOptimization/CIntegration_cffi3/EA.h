#ifndef EA_H
#define EA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Parameters for GPU-optimized EA
#define EA_POPULATION_SIZE 16384 // Increased for GPU saturation
#define EA_GENERATIONS 1000      // Kept for convergence
#define EA_DIM 10                // Matches Schwefel
#define EA_MU 5.0f               // Tighter crossover distribution
#define EA_MUM 10.0f             // Stronger mutation
#define EA_CROSSOVER_PROB 0.9f   // Crossover probability
#define EA_MUTATION_PROB 0.3f    // Increased mutation probability
#define EA_TOURNAMENT_SIZE 4     // For better parent selection

// Function declarations
void EA_optimize(Optimizer* opt, ObjectiveFunction objective_function);
void EA_free(Optimizer* opt);

#ifdef __cplusplus
}
#endif

#endif /* EA_H */
