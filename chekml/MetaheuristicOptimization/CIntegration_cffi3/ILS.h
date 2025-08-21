#ifndef ILS_H
#define ILS_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include "generaloptimizer.h"

// Optimization parameters
#define ILS_STEP_SIZE 0.05f
#define ILS_PERTURBATION_SIZE 1.0f

// ILS-specific function declaration
void ILS_optimize(Optimizer* opt, ObjectiveFunction objective_function);

#ifdef __cplusplus
}
#endif

#endif // ILS_H
