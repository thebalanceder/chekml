#ifndef CS_H
#define CS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define CS_MUTATION_PROBABILITY 0.1f
#define CS_CLONE_FACTOR 0.05f
#define CS_REPLACEMENT_RATE 0.1f

// Optimization execution
void CS_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // CS_H
