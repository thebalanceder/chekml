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

// Inline utility for fast random number generation (provided by GWO.c)
static inline double gwo_rand_double(void *rng_state);

// GWO Algorithm Phases
void gwo_initialize_population(Optimizer *opt, void *rng_state);
void update_hierarchy(Optimizer *opt, Solution *beta_solution, Solution *delta_solution);
void gwo_update_positions(Optimizer *opt, double a, Solution *beta_solution, Solution *delta_solution);

// Main Optimization Function
void GWO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // GWO_H
