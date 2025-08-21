#ifndef GLS_H
#define GLS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// GLS-specific parameters
#define GLS_POPULATION_SIZE 256
#define GLS_LOCAL_WORK_SIZE 64
#define GLS_LAMBDA 0.1f
#define GLS_PERTURBATION 0.1f

// GLS optimization function
void GLS_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // GLS_H
