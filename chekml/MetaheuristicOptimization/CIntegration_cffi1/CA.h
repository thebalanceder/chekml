#ifndef CRICKET_ALGORITHM_H
#define CRICKET_ALGORITHM_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure malloc/free work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define CA_POPULATION_SIZE 25
#define CA_TOL 1e-6
#define CA_MAX_ITER 100
#define CA_BETA_MIN 0.2
#define CA_Q_MIN 0.0
#define CA_PI 3.141592653589793

// ⚙️ Cricket Algorithm Constants
#define CA_TEMP_MIN 55.0
#define CA_TEMP_MAX 180.0
#define CA_TEMP_COEFF 0.891797
#define CA_TEMP_OFFSET 40.0252
#define CA_FREQ_OFFSET 10000.0

// 🌊 Cricket Algorithm Functions
double CA_rand_double(double min, double max);
double CA_CoefCalculate(double F, double T);
void CA_cricket_update_phase(Optimizer *opt);
void CA_optimize(Optimizer *opt, double (*objective_function)(double *));
void CA_optimize_with_history(Optimizer *opt, double (*objective_function)(double *), double **history, int *history_size, int max_history);

#ifdef __cplusplus
}
#endif

#endif // CRICKET_ALGORITHM_H
