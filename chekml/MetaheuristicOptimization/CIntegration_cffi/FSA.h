#ifndef FSA_H
#define FSA_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>  // For rand()
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define FSA_MAX_ITERATIONS 100
#define FSA_NUM_RUNS 30
#define FSA_DEBUG 0  // Set to 1 to enable logging

// FSA Algorithm Phases
void update_population_fsa(Optimizer *opt);
void update_with_initial_strategy(Optimizer *opt);

// Optimization Execution
void FSA_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // FSA_H
