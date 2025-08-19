#ifndef SASA_H
#define SASA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define SASA_POPULATION_SIZE 50 // Matches benchmark.py population_size
#define SASA_MAX_ITERATIONS 100 // Matches benchmark.py max_iter
#define SASA_DIM 2             // Matches benchmark.py dim
#define SASA_NUM_THREADS omp_get_max_threads() // Use all available CPU cores

// ⚙️ SSA Constants
#define SASA_C1_FACTOR 2.0
#define SASA_C1_EXPONENT 4.0

// 🚀 Optimization Execution
void SaSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SASA_H
