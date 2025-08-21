#ifndef POA_H
#define POA_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define STEP_SIZE 0.1

// 🌱 Utility function for random number generation
inline __attribute__((always_inline)) double rand_double_poa(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// 🌱 POA Algorithm Phases
void initialize_population_poa(Optimizer *restrict opt);
void evaluate_and_update_best(Optimizer *restrict opt, double (*objective_function)(double *));
void update_positions_poa(Optimizer *restrict opt);

// 🚀 Optimization Execution
void POA_optimize(Optimizer *restrict opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // POA_H
