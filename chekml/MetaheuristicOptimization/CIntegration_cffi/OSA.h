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

// 🌍 Utility function
static inline double rand_double_osa(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// 🦉 OSA Algorithm Phases
void osa_exploration_phase(Optimizer *restrict opt, int index, double *restrict bounds);
void osa_exploitation_phase(Optimizer *restrict opt, int index, const double *restrict best_pos, double *restrict bounds);

// 🚀 Optimization Execution
void OSA_optimize(Optimizer *restrict opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // OSA_H
