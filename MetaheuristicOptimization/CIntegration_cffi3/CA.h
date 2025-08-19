#ifndef CRICKET_ALGORITHM_H
#define CRICKET_ALGORITHM_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define CA_POPULATION_SIZE 25
#define CA_TOL 1e-6
#define CA_MAX_ITER 100
#define CA_BETA_MIN 0.2f
#define CA_Q_MIN 0.0f
#define CA_PI 3.141592653589793f

// Cricket Algorithm Constants
#define CA_TEMP_MIN 55.0f
#define CA_TEMP_MAX 180.0f
#define CA_TEMP_COEFF 0.891797f
#define CA_TEMP_OFFSET 40.0252f
#define CA_FREQ_OFFSET 10000.0f

// Cricket Algorithm Functions
void CA_optimize(Optimizer *opt, double (*objective_function)(double *));
void CA_optimize_with_history(Optimizer *opt, double (*objective_function)(double *), double **history, int *history_size, int max_history);

#ifdef __cplusplus
}
#endif

#endif // CRICKET_ALGORITHM_H
