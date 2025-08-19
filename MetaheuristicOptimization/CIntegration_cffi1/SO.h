#ifndef SO_H
#define SO_H

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
#define SPIRAL_STEP 0.1  // Controls the step size in spiral movement
#define TRIG_TABLE_SIZE 1024  // Size of precomputed trig table

// 🌊 SO Algorithm Phases
void initialize_population(Optimizer *opt);
void spiral_movement_phase(Optimizer *opt);
void update_and_sort_population(Optimizer *opt, double (*objective_function)(double *));

// 🚀 Optimization Execution
void SO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // SO_H
