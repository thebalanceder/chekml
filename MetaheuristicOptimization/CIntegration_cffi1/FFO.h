#ifndef FFO_H
#define FFO_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define SMELL_RANGE 1.0  // Range for random direction in smell-based search
#define SWARM_SIZE 20    // Default population size
#define MAX_ITER 100     // Default maximum iterations

// 🐝 FOA Algorithm Phases
void initialize_swarm(Optimizer *opt);
void smell_based_search(Optimizer *opt);
void evaluate_swarm(Optimizer *opt, double (*objective_function)(double *));
void vision_based_update(Optimizer *opt);

// 🚀 Optimization Execution
void FFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // FFO_H
