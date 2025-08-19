#ifndef FIREFLY_ALGORITHM_H
#define FIREFLY_ALGORITHM_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define FA_ALPHA 1.0
#define FA_BETA0 1.0
#define FA_GAMMA 0.01
#define FA_THETA 0.97
#define FA_POPULATION_SIZE 20
#define FA_MAX_ITER 500

// Firefly Algorithm Phases
void initialize_fireflies(Optimizer *opt, ObjectiveFunction objective_function, double **positions_out, int minimize);
void update_fireflies(Optimizer *opt, int t, ObjectiveFunction objective_function, int minimize);
void rank_fireflies(Optimizer *opt, int minimize);

// Optimization Execution
void FirefA_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // FIREFLY_ALGORITHM_H
