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
#define ENERGY_FACTOR 2.0  // Initial energy factor for exploration/exploitation

// 🌟 HHO Algorithm Phases
void exploration_phase(Optimizer *opt, double (*objective_function)(double *));
void exploitation_phase(Optimizer *opt, double (*objective_function)(double *));
void levy_flight(double *step, int dim);

// 🚀 Optimization Execution
void HHO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // HHO_H
