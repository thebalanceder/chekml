#ifndef HHO_H
#define HHO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"

// 🔧 HHO Parameters
#define HHO_BETA 1.5  // Levy flight parameter
#define ENERGY_FACTOR 2.0  // Initial energy factor

// 🌟 HHO Algorithm Phases
void exploration_phase(Optimizer *opt, double (*objective_function)(double *));
void exploitation_phase(Optimizer *opt, double (*objective_function)(double *));
void levy_flight(double *step, int dim, unsigned int *rng_state);

// 🚀 Optimization Execution
void HHO_optimize(Optimizer *opt, double (*objective_function)(double *));

// 🛠 Utility Function
static inline double hho_rand_double(double min, double max, unsigned int *rng_state);

#ifdef __cplusplus
}
#endif

#endif // HHO_H
