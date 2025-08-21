#ifndef LOSA_H
#define LOSA_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define LOSA_POPULATION_SIZE 50
#define LOSA_MAX_ITERATIONS 100
#define LOSA_STEP_SIZE 0.1

// LSA Algorithm Functions
void LoSA_initialize_population(Optimizer *opt);
void LoSA_update_positions(Optimizer *opt);

// Optimization Execution
void LoSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // LOSA_H
