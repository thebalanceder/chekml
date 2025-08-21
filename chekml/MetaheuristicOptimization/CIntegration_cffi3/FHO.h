#ifndef FHO_H
#define FHO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define MIN_FIREHAWKS 1
#define MAX_FIREHAWKS_RATIO 0.2f  // Max firehawks = ceil(population_size * MAX_FIREHAWKS_RATIO)
#define IR_MIN 0.0f
#define IR_MAX 1.0f

// Optimization execution
void FHO_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // FHO_H
