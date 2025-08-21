#ifndef GWO_H
#define GWO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// GWO-specific parameters
#define GWO_A_MAX 2.0  // Initial value of coefficient 'a'
#define GWO_A_MIN 0.0  // Final value of coefficient 'a'

// Inline utility function for random number generation
static inline double gwo_rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// GWO Algorithm Phases
void initialize_population_gwo(Optimizer *opt);
void update_hierarchy(Optimizer *opt, Solution *beta_solution, Solution *delta_solution);
void gwo_update_positions(Optimizer *opt, double a, Solution *beta_solution, Solution *delta_solution);

// Main Optimization Function
void GWO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // GWO_H
