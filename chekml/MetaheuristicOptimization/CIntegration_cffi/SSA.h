#ifndef SSA_H
#define SSA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define MAX_GLIDING_DISTANCE 1.11
#define MIN_GLIDING_DISTANCE 0.5
#define GLIDING_CONSTANT 1.9
#define NUM_FOOD_SOURCES 4
#define HICKORY_NUT_TREE 1
#define ACORN_NUT_TREE 3
#define NO_FOOD_TREES 46

// SSA context to hold pre-allocated buffers
typedef struct {
    int *tree_types;              // Tree types for each squirrel
    double *velocities;           // Velocity buffer for all squirrels
    double *pulse_flying_rates;   // Pulse flying rates
} SSAContext;

// SSA Algorithm Phases
void initialize_squirrels(Optimizer *opt, ObjectiveFunction objective_function, SSAContext *ctx);
void update_squirrels(Optimizer *opt, ObjectiveFunction objective_function, SSAContext *ctx);

// Optimization Execution
void SSA_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SSA_H
