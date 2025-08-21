#ifndef IWO_H
#define IWO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// Optimization parameters
#define INITIAL_POP_SIZE 10
#define MAX_POP_SIZE 25
#define MIN_SEEDS 1
#define MAX_SEEDS 5
#define EXPONENT 2.0
#define SIGMA_INITIAL 0.5
#define SIGMA_FINAL 0.001

// Utility functions
double rand_double_iwo(double min, double max);
double rand_normal_iwo();

// IWO Algorithm Phases
void initialize_population_iwo(Optimizer *opt);
void evaluate_population_iwo(Optimizer *opt, double (*objective_function)(double *));
double update_standard_deviation(int iteration, int max_iter);
void reproduction_phase(Optimizer *opt, double sigma, double (*objective_function)(double *));
void competitive_exclusion(Optimizer *opt, double (*objective_function)(double *));

// Main Optimization Function
void IWO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // IWO_H
