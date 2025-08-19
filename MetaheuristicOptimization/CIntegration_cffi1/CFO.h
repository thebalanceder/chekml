#ifndef CFO_H
#define CFO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h> // For SIMD intrinsics
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define ALPHA 0.1
#define ALIGNMENT 32 // For AVX2 (32-byte alignment)

// 🚀 CFO Algorithm Phases
void initialize_population_cfo(Optimizer *restrict opt) __attribute__((hot));
void central_force_update(Optimizer *restrict opt) __attribute__((hot));
void update_best_solution_cfo(Optimizer *restrict opt, double (*objective_function)(const double *)) __attribute__((hot));

// 🚀 Optimization Execution
void CFO_optimize(Optimizer *restrict opt, double (*objective_function)(const double *)) __attribute__((hot));

#ifdef __cplusplus
}
#endif

#endif // CFO_H
