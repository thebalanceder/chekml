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

// Utility function for random number generation
double rand_double(double min, double max);

// RDA Algorithm Phases
void update_position_phase(Optimizer *opt);

// Optimization Execution
void RDA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // RDA_H
