#ifndef ABC_H
#define ABC_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 ABC Algorithm Parameters
#define ABC_POPULATION_SIZE 100
#define ABC_MAX_ITER 200
#define ABC_ONLOOKER_RATIO 1.0
#define ABC_TRIAL_LIMIT_FACTOR 0.6
#define ABC_ACCELERATION_BOUND 1.0

// 🎲 Utility Functions
double abc_rand_double(double min, double max);
int abc_roulette_wheel_selection(double *probabilities, int size);

// 🐝 ABC Algorithm Phases
void employed_bee_phase(Optimizer *opt, double (*objective_function)(double *), int *trial_counters);
void onlooker_bee_phase(Optimizer *opt, double (*objective_function)(double *), int *trial_counters);
void scout_bee_phase(Optimizer *opt, double (*objective_function)(double *), int *trial_counters);
void abc_update_best_solution(Optimizer *opt, double (*objective_function)(double *));

// 🚀 Optimization Execution
void ABC_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // ABC_H
