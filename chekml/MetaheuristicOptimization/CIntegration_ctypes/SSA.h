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

// Utility function
double rand_double(double min, double max);

// SSA Algorithm Phases
void initialize_squirrels(Optimizer *opt, ObjectiveFunction objective_function, int *tree_types);
void update_squirrel_position(Optimizer *opt, int idx, ObjectiveFunction objective_function, int *tree_types);

// Optimization Execution
void SSA_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // SSA_H
