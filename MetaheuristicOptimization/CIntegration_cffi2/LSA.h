#ifndef LSA_H
#define LSA_H

#pragma once // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include <omp.h>     // ✅ For parallelization
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define MAX_CHANNEL_TIME 10
#define LSA_ENERGY_FACTOR 2.05
#define FOCKING_PROB 0.005   // Reduced for performance
#define DIRECTION_STEP 0.005

// 🌩️ LSA Algorithm Phases
void initialize_channels(Optimizer *opt, double *directions, int *channel_time);
void evaluate_channels(Optimizer *opt, ObjectiveFunction objective_function, int *best_idx);
void update_channel_elimination(Optimizer *opt, double *directions, int *channel_time);
void update_directions(Optimizer *opt, double *directions, ObjectiveFunction objective_function, int best_idx);
void lsa_update_positions(Optimizer *opt, double *directions, ObjectiveFunction objective_function, int t, int best_idx);

// 🚀 Optimization Execution
void LSA_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // LSA_H
