#ifndef LOCAL_SEARCH_H
#define LOCAL_SEARCH_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#ifndef STEP_SIZE
#define STEP_SIZE 1.0        // Larger perturbations
#endif
#define NEIGHBOR_COUNT 1024  // Increased for better exploration
#define MAX_ITER 100

// Local Search Algorithm Functions
void generate_random_solution(Optimizer *opt, double *solution);
void generate_neighbors(Optimizer *opt, const double *current_solution, double *neighbors);
void LS_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // LOCAL_SEARCH_H
