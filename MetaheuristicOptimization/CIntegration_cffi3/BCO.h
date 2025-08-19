#ifndef BCO_H
#define BCO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define BCO_CHEMOTAXIS_STEP_MAX 0.2f
#define BCO_CHEMOTAXIS_STEP_MIN 0.01f
#define BCO_ELIMINATION_RATIO 0.2f
#define BCO_REPRODUCTION_THRESHOLD 0.5f
#define BCO_MIGRATION_PROBABILITY 0.1f
#define BCO_COMMUNICATION_PROB 0.5f

// Optimization execution
void BCO_optimize(Optimizer* opt, double (*objective_function)(double*));

#ifdef __cplusplus
}
#endif

#endif // BCO_H
