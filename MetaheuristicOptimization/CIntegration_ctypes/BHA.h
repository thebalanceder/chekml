#ifndef BHA_H
#define BHA_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define BHA_POPULATION_SIZE 50
#define BHA_MAX_ITER 500

// Black Hole Algorithm Phases
void initialize_stars(Optimizer *opt);
void update_star_positions(Optimizer *opt);
void replace_with_better_black_hole(Optimizer *opt);
void new_star_generation(Optimizer *opt);

// Optimization Execution
void BHA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BHA_H
