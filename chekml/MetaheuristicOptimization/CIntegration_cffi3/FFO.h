#ifndef FFO_H
#define FFO_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define FFO_SMELL_RANGE 1.0f
#define FFO_SWARM_SIZE 20
#define FFO_MAX_ITER 100
#define FFO_LOCAL_WORK_SIZE 64

// Optimization execution
void FFO_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // FFO_H
