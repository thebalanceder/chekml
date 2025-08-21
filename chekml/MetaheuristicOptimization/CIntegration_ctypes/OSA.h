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
double rand_double(double min, double max);

// 🦉 OSA Algorithm Phases
void osa_exploration_phase(Optimizer *opt, int index);
void osa_exploitation_phase(Optimizer *opt, int index);

// 🚀 Optimization Execution
void OSA_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // OSA_H
