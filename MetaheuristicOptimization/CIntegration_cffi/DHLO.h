#ifndef DHLO_H
#define DHLO_H

#pragma once  // Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // Ensure malloc/free work properly
#include <stdio.h>   // For debugging/logging
#include "generaloptimizer.h"  // Include the main optimizer header

// Optimization parameters
#define INITIAL_LEADERS 30
#define TOLERANCE_PERCENT 5.0
#define A_MAX 2.0
#define A_MIN 0.0
#define PSO_INERTIA 0.7
#define PSO_COGNITIVE 1.5
#define PSO_SOCIAL 1.5
#define RW_STEP_SCALE 0.1

// DHLO configuration
#define DHLO_VARIANT "V4"  // Options: V1, V2, V3, V4
#define POSITION_UPDATE_STRATEGY "GWO"  // Options: GWO, PSO, RW

// Inline function to check bounds with logging
static inline void check_bounds(double *position, const double *bounds, int dim) {
    for (int i = 0; i < dim; i++) {
        double lb = bounds[2 * i];
        double ub = bounds[2 * i + 1];
        double old_pos = position[i];
        position[i] = fmax(lb, fmin(ub, position[i]));
        if (position[i] != old_pos) {
            fprintf(stderr, "Clamped position[%d] from %f to %f (lb=%f, ub=%f)\n", i, old_pos, position[i], lb, ub);
        }
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
