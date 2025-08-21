#ifndef PRO_H
#define PRO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#ifndef POPULATION_SIZE
#define PRO_POPULATION_SIZE 30
#else
#define PRO_POPULATION_SIZE POPULATION_SIZE
#endif
#define PRO_MAX_EVALUATIONS 2000  // Reduced for 2D benchmark functions
#define REINFORCEMENT_RATE 0.7
#define SCHEDULE_MIN 0.9
#define SCHEDULE_MAX 1.0

// Top-k selection for schedules
void select_top_k(double *arr, int *indices, int n, int k);

// PRO Algorithm Phases
void select_behaviors(Optimizer *opt, int i, int current_eval, double **schedules, int *selected_behaviors, int *landa);
void stimulate_behaviors(Optimizer *opt, int i, int *selected_behaviors, int landa, int k, int current_eval, double **schedules, double *new_solution);
void apply_reinforcement(Optimizer *opt, int i, int *selected_behaviors, int landa, double **schedules, double *new_solution, double new_fitness);
void reschedule(Optimizer *opt, int i, double **schedules);

// Optimization Execution
void PRO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // PRO_H
