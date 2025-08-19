#ifndef DHLO_H
#define DHLO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include "generaloptimizer.h"

// Optimization parameters
#define INITIAL_LEADERS 20
#define TOLERANCE_PERCENT 5.0
#define A_MAX 2.0
#define A_MIN 0.0
#define PSO_INERTIA 0.7
#define PSO_COGNITIVE 1.5
#define PSO_SOCIAL 1.5
#define RW_STEP_SCALE 0.1
#define DHLO_MAX_DIM 2
#define DHLO_MAX_POP 30
#define DHLO_MAX_ITER 300

// DHLO configuration
#define DHLO_VARIANT "V4"
#define POSITION_UPDATE_STRATEGY "GWO"

// Inline bound checking
static inline void check_bounds(double *restrict position, const double *restrict bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        double lb = bounds[2 * i];
        double ub = bounds[2 * i + 1];
        position[i] = fmax(lb, fmin(ub, position[i]));
    }
}

// DHLO Algorithm Phases
int dhlo_initialize_population(Optimizer *opt, int num_leaders, Solution **leaders, Solution **pbest);
int dhlo_update_leaders(Optimizer *opt, int iter, double *leader_fitness_history, int num_leaders, Solution *leaders, Solution *pbest);
int dhlo_adjust_num_leaders(Optimizer *opt, int iter, double *leader_fitness_history, int *num_leaders);
int dhlo_update_positions(Optimizer *opt, int iter, int num_leaders, Solution *leaders, Solution *pbest, double *temp_pos, double *velocity);

// Optimization Execution
int DHLO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // DHLO_H
