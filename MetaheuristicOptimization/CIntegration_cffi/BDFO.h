#ifndef BDFO_H
#define BDFO_H

#pragma once  // ✅ Improves header inclusion efficiency

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>  // ✅ Ensure rand/RAND_MAX work properly
#include <stdio.h>   // ✅ For debugging/logging
#include "generaloptimizer.h"  // ✅ Include the main optimizer header

// 🔧 Optimization parameters
#define BDFO_EXPLORATION_FACTOR 0.5
#define BDFO_ADJUSTMENT_RATE 0.3
#define BDFO_ELIMINATION_RATIO 0.23
#define BDFO_PERTURBATION_SCALE 0.1
#define BDFO_EXPLORATION_PROB 0.3
#define BDFO_SEGMENT_FACTOR 0.5  // Split solutions into two segments

// ⚙️ Inline utility functions
static inline double bdfo_rand_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// 🐬 BDFO Algorithm Phases
void bdfo_bottom_grubbing_phase(Optimizer *opt, double (*objective_function)(double *));
void bdfo_exploration_phase(Optimizer *opt);
void bdfo_elimination_phase(Optimizer *opt);

// 🚀 Optimization Execution
void BDFO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // BDFO_H
