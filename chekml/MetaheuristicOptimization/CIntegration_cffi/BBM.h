#ifndef BBM_H
#define BBM_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "generaloptimizer.h"

// Optimization parameters
#define QUEEN_FACTOR 0.3
#define DRONE_SELECTION 0.2
#define WORKER_IMPROVEMENT 1.35
#define BROOD_DISTRIBUTION 0.46
#define MATING_RESISTANCE 1.2
#define REPLACEMENT_RATIO 0.23
#define CFR_FACTOR 9.435
#define BLEND_ALPHA 0.5

// Function prototypes
double rand_double_bbm(double min, double max);
double rand_gamma(double shape, double scale);
void initialize_bees(Optimizer *opt, double *temp_buffer);
void queen_selection_phase(Optimizer *opt, double (*objective_function)(double *));
void blend_alpha_crossover(double *queen, double *drone, double *new_solution, int dim);
void mating_phase(Optimizer *opt, double (*objective_function)(double *), double *new_solution, double *temp_solution);
void worker_phase(Optimizer *opt, double (*objective_function)(double *), double *new_solution);
void replacement_phase(Optimizer *opt, double (*objective_function)(double *));
void BBM_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BBM_H
