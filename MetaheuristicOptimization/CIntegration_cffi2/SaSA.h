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
#define SASA_NUM_THREADS 4     // Number of OpenMP threads for parallelization

// ⚙️ SSA Constants
#define SASA_C1_FACTOR 2.0
#define SASA_C1_EXPONENT 4.0

// 🌊 SSA Algorithm Phases
void sasa_initialize_population(Optimizer *opt);
void sasa_update_positions(Optimizer *opt, double c1);
void sasa_evaluate_fitness(Optimizer *opt, double (*objective_function)(double *));

// 🚀 Optimization Execution
void SaSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SASA_H
