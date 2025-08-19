#ifndef TEO_H
#define TEO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define STEP_SIZE 0.1
#define INITIAL_TEMPERATURE 100.0
#define FINAL_TEMPERATURE 0.01
#define COOLING_RATE 0.99

// TEO Algorithm Phases
void perturb_solution(Optimizer *opt, double *new_solution);
void accept_solution(Optimizer *opt, double *new_solution, double new_fitness, double temperature);
void update_best_solution_teo(Optimizer *opt);
void cool_temperature(double *temperature);

// Optimization Execution
void TEO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // TEO_H
