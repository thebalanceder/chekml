#ifndef OSA_H
#define OSA_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define OSA_STEP_SIZE 0.1
#define OSA_P_EXPLORE 0.1
#ifndef INV_RAND_MAX
#define INV_RAND_MAX (1.0 / RAND_MAX)
#endif

// 🌍 Fast PRNG for OSA
static inline double osa_fast_rand(double min, double max, unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return min + (max - min) * ((double)(*seed & 0x7fffffff) * INV_RAND_MAX);
}

// 🦉 OSA Algorithm Phases
void osa_exploration_phase(Optimizer *restrict opt, int index, double *restrict bounds, unsigned int *seed);
void osa_exploitation_phase(Optimizer *restrict opt, int index, const double *restrict best_pos, double *restrict bounds, unsigned int *seed);

// 🚀 Optimization Execution
void OSA_optimize(Optimizer *restrict opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // OSA_H
