#ifndef ALO_H
#define ALO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "generaloptimizer.h"
#include <CL/cl.h>

// Optimization parameters
#define I_FACTOR_1 100.0f      // Scaling factor for I after 10% iterations
#define I_FACTOR_2 1000.0f     // Scaling factor for I after 50% iterations
#define I_FACTOR_3 10000.0f    // Scaling factor for I after 75% iterations
#define I_FACTOR_4 100000.0f   // Scaling factor for I after 90% iterations
#define I_FACTOR_5 1000000.0f  // Scaling factor for I after 95% iterations
#define ROULETTE_EPSILON 1e-10f // Small value to avoid division by zero

// Optimization Execution
void ALO_optimize(Optimizer *opt, double (*objective_function)(double *));

#ifdef __cplusplus
}
#endif

#endif // ALO_H
