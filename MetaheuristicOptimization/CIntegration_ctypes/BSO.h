#ifndef BSO_H
#define BSO_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define PERTURBATION_SCALE_BSO 0.1  // Scale for random perturbation in local search

// BSO Algorithm Phases
void initialize_population_bso(Optimizer *opt);
void local_search(Optimizer *opt);

// Optimization Execution
void BSO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BSO_H
