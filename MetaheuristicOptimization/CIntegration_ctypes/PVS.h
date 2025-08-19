#ifndef PVS_H
#define PVS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 Optimization parameters
#define DISTRIBUTION_INDEX 20
#define X_GAMMA 0.1

// 🚀 PVS Algorithm Phases
void initialize_vortex(Optimizer *opt);
void first_phase(Optimizer *opt, int iteration, double radius);
void second_phase(Optimizer *opt, int iteration, ObjectiveFunction objective_function);
void polynomial_mutation(Optimizer *opt, double *solution, double *mutated, int *state);

// 🌌 Optimization Execution
void PVS_optimize(Optimizer *opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // PVS_H
