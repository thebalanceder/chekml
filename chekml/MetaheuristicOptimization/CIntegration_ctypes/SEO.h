#ifndef SEO_H
#define SEO_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define SEO_POPULATION_SIZE 50
#define SEO_MAX_ITER 100

// SEO Algorithm Phases
void social_engineering_update(Optimizer *opt);

// Optimization Execution
void SEO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SEO_H
