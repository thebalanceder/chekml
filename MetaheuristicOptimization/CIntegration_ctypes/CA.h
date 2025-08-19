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
#define POPULATION_SIZE_CA 25
#define ALPHA_CA 0.5
#define BETA_MIN 0.2
#define Q_MIN 0.0
#define TOL 1e-6
#define MAX_ITER_CA 100
#define PI 3.14

// ⚙️ Cricket Algorithm Constants
#define TEMP_MIN 55.0
#define TEMP_MAX 180.0
#define TEMP_COEFF 0.891797
#define TEMP_OFFSET 40.0252
#define FREQ_OFFSET 10000.0

// 🌊 Cricket Algorithm Functions
double CoefCalculate(double F, double T);
void cricket_update_phase(Optimizer *opt);
void CA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // CRICKET_ALGORITHM_H
