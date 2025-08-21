#ifndef RMO_H
#define RMO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define POPULATION_SIZE 50
#define MAX_ITERATIONS 100
#define ALPHA 0.1

// 🌟 RMO Algorithm Phases
// Optimized for extreme CPU performance with OpenMP, SIMD, and minimal memory overhead
void rmo_initialize_population(Optimizer *opt);
void rmo_evaluate_population(Optimizer *opt, double (*objective_function)(double *));
void rmo_sort_population(Optimizer *opt);
void update_reference_point_and_positions(Optimizer *opt, double *reference_point);

// 🚀 Optimization Execution
void RMO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // RMO_H
