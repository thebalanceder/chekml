#ifndef BCO_H
#define BCO_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define CHEMOTAXIS_STEP_MAX 0.2
#define CHEMOTAXIS_STEP_MIN 0.01
#define ELIMINATION_RATIO_BCO 0.2
#define REPRODUCTION_THRESHOLD 0.5
#define MIGRATION_PROBABILITY 0.1
#define COMMUNICATION_PROB 0.5

// BCO Algorithm Phases
void chemotaxis_and_communication(Optimizer *opt, int iteration);
void elimination_and_reproduction(Optimizer *opt);
void migration_phase(Optimizer *opt);

// Optimization Execution
void BCO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BCO_H
