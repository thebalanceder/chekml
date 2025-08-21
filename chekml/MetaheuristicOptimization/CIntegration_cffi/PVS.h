#ifndef PVS_H
#define PVS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define DISTRIBUTION_INDEX 20
#define X_GAMMA 0.1

// 🌌 PVS-specific data
typedef struct {
    double *center; // Vortex center
    double *obj_vals; // Fitness values
    double *prob; // Roulette wheel probabilities
    double *sol; // Temporary solution for crossover
    double *mutated; // Temporary solution for mutation
    double prob_mut; // Mutation probability
    double prob_cross; // Crossover probability
    double *bound_diffs; // Precomputed (upper - lower) bounds
} PVSData;

// 🚀random double generation
double rand_double(double min, double max);

// 🚀 PVS Algorithm Phases
void initialize_vortex(Optimizer *opt, PVSData *data);
void first_phase(Optimizer *opt, PVSData *data, int iteration, double radius);
void second_phase(Optimizer *opt, PVSData *data, int iteration, ObjectiveFunction objective_function);
void polynomial_mutation(Optimizer *opt, PVSData *data, double *solution, double *mutated, int *state);

// 🌌 Optimization Execution
void PVS_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // PVS_H
