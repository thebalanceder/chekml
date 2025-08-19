#ifndef RDA_H
#define RDA_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define STEP_SIZE 0.1
#define P_EXPLORATION 0.1
#define INV_RAND_MAX (1.0 / RAND_MAX)  // Precompute for faster random number generation

// Utility function for random number generation (inline for performance)
static inline double rand_double_rda(double min, double max) {
    return min + (max - min) * (double)rand() * INV_RAND_MAX;
}

// RDA Algorithm Phases (main phase inlined in RDA.c for performance)
void RDA_optimize(Optimizer *restrict opt, double (*objective_function)(double *restrict));

#ifdef __cplusplus
}
#endif

#endif // RDA_H
