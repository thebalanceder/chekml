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
#define DHLO_INITIAL_LEADERS 30
#define DHLO_TOLERANCE_PERCENT 5.0
#define DHLO_A_MAX 2.0
#define DHLO_A_MIN 0.0
#define DHLO_PSO_INERTIA 0.7
#define DHLO_PSO_COGNITIVE 1.5
#define DHLO_PSO_SOCIAL 1.5
#define DHLO_RW_STEP_SCALE 0.1
#define DHLO_MAX_DIM 2
#define DHLO_MAX_POP 30
#define DHLO_MAX_ITER 200

// DHLO configuration (change these to select variant/strategy)
#define DHLO_VARIANT "V4"  // Options: V1, V2, V3, V4
#define POSITION_UPDATE_STRATEGY "GWO"  // Options: GWO, PSO, RW

// Inline bound checking
static inline void check_bounds(double *restrict position, const double *restrict bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        double lb = bounds[2 * i];
        double ub = bounds[2 * i + 1];
        position[i] = fmax(lb, fmin(ub, position[i]));
    }
}

// DHLO Algorithm Phases
void dhlo_initialize_population(Optimizer *opt, int num_leaders, Solution **leaders, Solution **pbest);
void dhlo_update_leaders(Optimizer *opt, int iter, double *leader_fitness_history, int num_leaders, Solution *leaders, Solution *pbest);
void dhlo_adjust_num_leaders(Optimizer *opt, int iter, double *leader_fitness_history, int *num_leaders);
void dhlo_update_positions(Optimizer *opt, int iter, int num_leaders, Solution *leaders, Solution *pbest, double *temp_pos, double *velocity);
void DHLO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // DHLO_H
