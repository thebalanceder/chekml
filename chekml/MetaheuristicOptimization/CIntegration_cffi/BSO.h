#ifndef BSO_H
#define BSO_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // For memory management
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Main optimizer header

// Optimization parameters
#define PERTURBATION_SCALE 0.1  // Scale for random perturbation in local search

// Macro for bounds clipping
#define CLIP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

// Fast inline random number generation
static inline double rand_double_bso(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// BSO Algorithm Phases
void initialize_population_bso(Optimizer *opt);
void local_search(Optimizer *opt);

// Optimization Execution
void BSO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BSO_H
