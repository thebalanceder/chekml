#pragma once

#ifndef SDS_H
#define SDS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters (tuned for GPU)
#define SDS_MUTATION_RATE 0.1f          // Mutation probability
#define SDS_MUTATION_SCALE 3.0f         // Mutation scale
#define SDS_CLUSTER_THRESHOLD 0.4f      // Convergence threshold
#define SDS_CONVERGENCE_TOLERANCE 1e-3f // Distance threshold for clustering
#define SDS_MAX_COMPONENTS 10           // Maximum number of component functions
#define SDS_INV_MUTATION_SCALE (1.0f / SDS_MUTATION_SCALE) // Precomputed reciprocal
#define SDS_STAGNATION_TOLERANCE 1e-6f  // Stagnation check threshold

// Function declaration
void SDS_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SDS_H
