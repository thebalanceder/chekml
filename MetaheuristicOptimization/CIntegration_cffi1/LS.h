#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure basic utilities
#include <stdio.h>   // ✅ For error logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define STEP_SIZE 0.1        // Controls the magnitude of local search steps
#define NEIGHBOR_COUNT 10    // Number of neighbors per iteration
#define MAX_ITER 100         // Maximum number of iterations
#define MAX_DIM 10           // Maximum dimensionality for stack allocation
#define CACHE_LINE 64        // Cache line size for alignment

// 🌟 Local Search Algorithm Function
void LS_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // LOCAL_SEARCH_H
